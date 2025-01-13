import static
import torch
import math
import torch_pruning as tp
import torch.nn.functional as F

from typing import Sequence
from torch import nn
from timm.models.vision_transformer import Attention
from peft import get_peft_model, LoraConfig

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


    
def linear_freeze_forward(self, x):
    out = nn.functional.linear(x, self.weight, self.bias)

    if hasattr(self, 'prune_in_mask'):
        x = x[..., self.prune_in_mask]

    prune_out = nn.functional.linear(x, self.prune_weight, self.prune_bias)

    if hasattr(self, 'prune_out_mask'):
        out[..., self.prune_out_mask] += prune_out
    else:
        out += prune_out

    return out

def conv_freeze_forward(self, x):
    out = nn.functional.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dialtion) # no groups pls

    if hasattr(self, 'prune_in_mask'):
        x = x[:, self.prune_in_mask]

    prune_out = nn.functional.conv2d(x[self.prune_in_mask], self.prune_weight, self.prune_bias, self.stride, self.padding, self.dialtion)

    if hasattr(self, 'prune_out_mask'):
        out[:, self.prune_out_mask] += prune_out
    else:
        out += prune_out

    return out

class FreezePruner(tp.BasePruningFunc):
    def _init_layer(self, layer: nn.Module):
        if not hasattr(layer, 'prune_weight'):
            layer.weight.requires_grad = False
            layer.prune_weight = nn.Parameter(torch.zeros(layer.weight.shape), requires_grad=True)

            if layer.bias:
                layer.bias.requires_grad = False
                layer.prune_bias = nn.Parameter(torch.zeros(layer.bias.shape), requires_grad=True)
            
            if isinstance(layer, nn.Linear):
                layer.forward = linear_freeze_forward.__get__(layer, nn.Linear)
            elif isinstance(layer, nn.Conv2d):
                layer.forward = conv_freeze_forward.__get__(layer, nn.Conv2d)

    def prune_out_channels(self, layer: nn.Module, idxs: list):
        self._init_layer(layer)
        layer.prune_out_mask = torch.ones(layer.weight.size(0), dtype=torch.bool)
        layer.prune_out_mask[idxs] = False
        layer.prune_weight = nn.Parameter(layer.prune_weight[layer.prune_out_mask], requires_grad=True)

        if layer.bias:
            layer.prune_bias = nn.Parameter(layer.prune_bias[layer.prune_out_mask], requires_grad=True)

        layer.out_features = int(layer.prune_out_mask.sum())

        return layer

    def prune_in_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        self._init_layer(layer)
        layer.prune_in_mask = torch.ones(layer.weight.size(1), dtype=torch.bool)
        layer.prune_in_mask[idxs] = False
        layer.prune_weight = nn.Parameter(layer.prune_weight[:, layer.prune_in_mask], requires_grad=True) # Does not work with conv groups

        layer.in_features = int(layer.prune_in_mask.sum())

        return layer
    
    def get_out_channels(self, layer):
        pass

    def get_in_channels(self, layer):
        pass
