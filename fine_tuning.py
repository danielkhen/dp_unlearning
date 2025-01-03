import static
import torch
import math
import torch_pruning as tp

from torch import nn

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

# def get_pruning_mask(param, amount):
#     with torch.no_grad():
#         flattened_weights = param.abs().view(-1)
#         threshold_index = int(len(flattened_weights) * amount)
#         threshold_value = torch.sort(flattened_weights)[0][threshold_index]

#         return (param.abs() >= threshold_value)

# def prune_gradients(target_children, amount):
#     target_parameters = [module.weight for child in target_children 
#                         for module in child.modules() 
#                         if isinstance(module, PEFT_SUPPORTED_TYPES)]
    
#     for param in target_parameters:
#         mask = get_pruning_mask(param, amount=amount).to(static.CUDA)
#         param.register_hook(lambda grad, mask=mask: grad * mask) # Ensure mask scope is in lambda function

# def prune_model(target_children, amount):
#     target_modules = [module for child in target_children 
#                         for module in child.modules() 
#                         if isinstance(module, PEFT_SUPPORTED_TYPES)]
    
#     for module in target_modules:
#         prune.l1_unstructured(module, 'weight', amount=amount)

def prune(model, target_modules, importance, importance_kwargs={}):
    importance = getattr(tp.importance, importance)
    pruning_ratio_dict = {module: 1 - 1/math.sqrt(peft_ratio) for _, module, peft_ratio in target_modules}
    ignored_layers = set(name for name, _ in model.named_modules()) - set(name for name, _, _ in target_modules)
    
    pruner = tp.pruner.MetaPruner(
        model,
        torch.randn(1, 3, static.IMG_SIZE, static.IMG_SIZE),
        importance=importance(**importance_kwargs),
        pruning_ratio_dict=pruning_ratio_dict,
        ignored_layers=ignored_layers
    )

    pruner.step()


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