import static
import torch
import math
import torch_pruning as tp
import torch.nn.functional as F

from typing import Sequence
from torch import nn
from timm.models.vision_transformer import Attention
from peft import get_peft_model, LoraConfig
from torch.nn.utils import parametrize
def unfreeze_peft_model(model):
    for name, module in model.named_modules():
        if name.endswith('.base_layer'): 
            continue # If not original layer before lora

        for param_name, param in module.named_parameters(): 
            if '.' in param_name:
                continue # Only parameters of self

            param.requires_grad = True

def peft_ratio_to_rank(module, ratio):
    if isinstance(module, nn.Linear):
        return min(module.in_features, module.out_features) // ratio
    elif isinstance(module, nn.Conv2d):
        return min(module.in_channels, module.out_channels) // ratio


def get_lora_model(model, target_modules, lora_alpha, lora_dropout):
    rank_pattern = {name: peft_ratio_to_rank(module, peft_ratio) for name, module, peft_ratio in target_modules}
    target_modules = [name for name, _, _ in target_modules]
    model = get_peft_model(model, LoraConfig(target_modules=target_modules, rank_pattern=rank_pattern, lora_alpha=lora_alpha, lora_dropout=lora_dropout))
    unfreeze_peft_model(model)

    return model

def attention_forward(self, x):
    """https://github.com/huggingface/pytorch-image-models/blob/054c763fcaa7d241564439ae05fbe919ed85e614/timm/models/vision_transformer.py#L79"""
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)

    if self.fused_attn:
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p,
        )
    else:
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

    x = x.transpose(1, 2).reshape(B, N, -1) # original implementation: x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x

def peft_ratio_to_pruning_ratio(x, linear=False):
    if not linear:
        x = math.sqrt(x)

    return 1 - 1/x

def prune(model, target_modules, ignored_layers, importance, global_pruning=False, importance_kwargs={}, freeze=True):
    num_heads = {}

    for _, module, _ in target_modules:
        if isinstance(module, Attention):
            module.forward = attention_forward.__get__(module, Attention) # https://stackoverflow.com/questions/50599045/python-replacing-a-function-within-a-class-of-a-module
            num_heads[module.qkv] = module.num_heads 
    
    importance = getattr(tp.importance, importance)
    pruning_ratio_dict = {} if global_pruning else {module: peft_ratio_to_pruning_ratio(peft_ratio, linear=bool(num_heads)) for _, module, peft_ratio in target_modules}
    pruning_ratio = peft_ratio_to_pruning_ratio(target_modules[0][2], linear=bool(num_heads))

    pruner = tp.pruner.MetaPruner(
        model,
        torch.randn(1, 3, static.IMG_SIZE, static.IMG_SIZE),
        importance=importance(**importance_kwargs),
        pruning_ratio=pruning_ratio,
        pruning_ratio_dict=pruning_ratio_dict,
        ignored_layers=ignored_layers,
        global_pruning=global_pruning,
        num_heads=num_heads,
        customized_pruners={nn.Conv2d: FreezePruner(), nn.Linear: FreezePruner()}
    )
            
    if freeze:
        for group in pruner.step(interactive=True):
            for dep, idxs in group:
                layer = dep.target.module
                pruning_fn = dep.handler
                freeze_pruner = FreezePruner()

                if pruning_fn in [tp.prune_conv_in_channels, tp.prune_linear_in_channels]:
                    freeze_pruner.prune_in_channels(layer, idxs)
                elif pruning_fn in [tp.prune_conv_out_channels, tp.prune_linear_out_channels]:
                    freeze_pruner.prune_out_channels(layer, idxs)
                # else:
                #     pruning_fn(layer, idxs)
                              
    else:
        pruner.step()

        for _, module, _ in target_modules:
            if isinstance(module, Attention):
                module.num_heads = pruner.num_heads[module.qkv]
                module.head_dim = module.qkv.out_features // (3 * module.num_heads)
    
# Replaces blocks directly under model
def replace_blocks(model, block_class, **kwargs):
    for name, block in model.named_children():
        setattr(model, name, block_class(block, **kwargs))

        for param in block.parameters():
            param.requires_grad = False

# Target children are assumed to contain the blocks to replace
def replace_modules(model, module_class, class_to_replace, **kwargs):
    for name, block in model.named_children():
        if isinstance(block, class_to_replace):
            setattr(model, name, module_class(block, **kwargs))

            for param in block.parameters():
                param.requires_grad = False
        else:
            replace_modules(block, module_class, class_to_replace, **kwargs)

def replace_module(model, target, module_cls, freeze=True, args_lambda=lambda _: [], kwargs_lambda=lambda _: {}):
    target_split = target.split('.')
    parent_module = model

    for name in target_split[:-1]:
        parent_module = getattr(parent_module, name)

    name = target_split[-1]
    module = getattr(parent_module, name)
    
    setattr(parent_module, target_split[-1], module_cls(*args_lambda(module), **kwargs_lambda(module)))

    if freeze:
        for param in module.parameters():
            param.requires_grad = False


class FreezeWeightParameterization(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
        self.weight = nn.Parameter(torch.zeros(shape), requires_grad=True)
        self.set_in_idxs([])
        self.set_out_idxs([])

    def forward(self, X):
        res = X.clone()

        if len(self.in_idxs) == self.shape[1] and len(self.out_idxs) == self.shape[0]:
            res += self.weight
        else:
            res[self.out_idxs.unsqueeze(1), self.in_idxs.unsqueeze(0)] += self.weight

        return res
    
    def set_in_idxs(self, idxs):
        idxs = set(range(self.shape[1])) - set(idxs)
        self.in_idxs = nn.Parameter(torch.tensor(list(idxs)), requires_grad=False)
        self.weight = nn.Parameter(self.weight[:, self.in_idxs], requires_grad=True)

    def set_out_idxs(self, idxs):
        idxs = set(range(self.shape[0])) - set(idxs)
        self.out_idxs = nn.Parameter(torch.tensor(list(idxs)), requires_grad=False)
        self.weight = nn.Parameter(self.weight[self.out_idxs], requires_grad=True)
    
class FreezeBiasParameterization(nn.Module):
    def __init__(self, len):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(len), requires_grad=True)
        self.set_out_idxs([])

    def forward(self, X):
        res = X.clone()

        if len(self.out_idxs) == self.len:
            res += self.bias
        else:
            res[self.out_idxs] += self.bias

        return res
    
    def set_out_idxs(self, idxs):
        idxs = set(range(self.len)) - set(idxs)
        self.out_idxs = nn.Parameter(torch.tensor(list(idxs)), requires_grad=False)
        self.bias = nn.Parameter(self.bias[self.out_idxs], requires_grad=True)


class FreezePruner(tp.BasePruningFunc):
    def _init_layer(self, layer: nn.Module):
        if not hasattr(layer, 'parametrizations'):
            layer.weight.requires_grad = False
            parametrize.register_parametrization(layer, 'weight', FreezeWeightParameterization(layer.weight.shape))

            if layer.bias:
                layer.bias.requires_grad = False
                parametrize.register_parametrization(layer, 'bias', FreezeBiasParameterization(layer.bias.size(0)))

    def prune_out_channels(self, layer: nn.Module, idxs: list):
        self._init_layer(layer)
        layer.parametrizations.weight[0].set_out_idxs(idxs)

        if layer.bias:
            layer.parametrizations.bias[0].set_out_idxs(idxs)

        # if isinstance(layer, nn.Conv2d):
        #     layer.out_channels = int(mask.sum())
        # elif isinstance(layer, nn.Linear):
        #     layer.out_features = int(mask.sum())

        return layer

    def prune_in_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        self._init_layer(layer)
        layer.parametrizations.weight[0].set_in_idxs(idxs)

        # if isinstance(layer, nn.Conv2d):
        #     layer.in_channels = int(mask.sum())
        # elif isinstance(layer, nn.Linear):
        #     layer.in_features = int(mask.sum())

        return layer
    
    def get_out_channels(self, layer):
        pass

    def get_in_channels(self, layer):
        pass
