import static
import torch

from peft import get_peft_model, LoraConfig
from torch.nn.utils import prune
from torch.nn import Linear, Conv2d

PEFT_SUPPORTED_TYPES = (Linear, Conv2d)

def unfreeze_peft_model(model):
    for name, module in model.named_modules():
        if name.endswith('.base_layer'): 
            continue # If not original layer before lora

        for param_name, param in module.named_parameters(): 
            if '.' in param_name:
                continue # Only parameters of self

            param.requires_grad = True

def get_lora_model(model, target_children, rank, lora_alpha, lora_dropout):
    target_modules = [module_name for child in target_children 
                        for module_name, module in child.named_modules() 
                        if isinstance(module, PEFT_SUPPORTED_TYPES)]
    
    model = get_peft_model(model, LoraConfig(target_modules=target_modules, r=rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout))
    unfreeze_peft_model(model)

    return model

def get_pruning_mask(param, amount):
    with torch.no_grad():
        flattened_weights = param.abs().view(-1)
        threshold_index = int(len(flattened_weights) * amount)
        threshold_value = torch.sort(flattened_weights)[0][threshold_index]

        return (param.abs() >= threshold_value)

def prune_gradients(target_children, amount):
    target_parameters = [module.weight for child in target_children 
                        for module in child.modules() 
                        if isinstance(module, PEFT_SUPPORTED_TYPES)]
    
    for param in target_parameters:
        mask = get_pruning_mask(param, amount=amount).to(static.CUDA)
        param.register_hook(lambda grad, mask=mask: grad * mask) # Ensure mask scope is in lambda function

def prune_model(target_children, amount):
    target_modules = [module for child in target_children 
                        for module in child.modules() 
                        if isinstance(module, PEFT_SUPPORTED_TYPES)]
    
    for module in target_modules:
        prune.l1_unstructured(module, 'weight', amount=amount)

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