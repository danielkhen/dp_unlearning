import static
import torch

from peft import get_peft_model, LoraConfig
from torch.nn.utils import prune
from torch.nn import Linear, Conv2d

PEFT_SUPPORTED_TYPES = (Linear, Conv2d)

def get_trainable_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)

def unfreeze_peft_model(model):
    for name, module in model.named_modules():
        if name.endswith('.base_layer'): 
            continue # If not original layer before lora

        for param_name, param in module.named_parameters(): 
            if '.' in param_name:
                continue # Only parameters of self

            param.requires_grad = True

def get_lora_model(model, target_children, rank, lora_alpha, lora_dropout):
    target_modules = [module_name for child_name in target_children 
                        for module_name, module in getattr(model, child_name).named_modules() 
                        if isinstance(module, PEFT_SUPPORTED_TYPES)]
    
    model = get_peft_model(model, LoraConfig(target_modules=target_modules, r=rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout))
    unfreeze_peft_model(model)

    return model, get_trainable_parameters(model)

def get_pruning_mask(param, amount):
    with torch.no_grad():
        flattened_weights = param.abs().view(-1)
        threshold_index = int(len(flattened_weights) * amount)
        threshold_value = torch.sort(flattened_weights)[0][threshold_index]

        return (param.abs() >= threshold_value)

def prune_gradients(model, target_children, amount):
    target_parameters = [module.weight for child_name in target_children 
                        for module in getattr(model, child_name).modules() 
                        if isinstance(module, PEFT_SUPPORTED_TYPES)]
    
    target_parameters_delta = int(sum(param.numel() for param in target_parameters) * amount)
    
    for param in target_parameters:
        mask = get_pruning_mask(param, amount=amount).to(static.CUDA)
        param.register_hook(lambda grad, mask=mask: grad * mask) # Ensure mask scope is in lambda function

    return get_trainable_parameters(model) - target_parameters_delta

def prune_model(model, target_children, amount):
    target_modules = [module for child_name in target_children 
                        for module in getattr(model, child_name).modules() 
                        if isinstance(module, PEFT_SUPPORTED_TYPES)]
    
    target_parameters_delta = int(sum(module.weight.numel() for module in target_modules) * amount)
    
    for module in target_modules:
        prune.l1_unstructured(module, 'weight', amount=amount)

    return get_trainable_parameters(model) - target_parameters_delta

# Target children are assumed to contain the blocks to replace
def replace_blocks(model, target_children, block_class, **kwargs):
    for child_name in target_children:
        child = getattr(model, child_name)

        for name, block in child.named_children():
            setattr(child, name, block_class(block, **kwargs))

            for param in block.parameters():
                param.requires_grad = False

    return get_trainable_parameters(model)

# Target children are assumed to contain the blocks to replace
def replace_modules(model, target_children, module_class, **kwargs):
    for child_name in target_children:
        child = getattr(model, child_name)

        for name, block in child.named_children():
            setattr(child, name, module_class(block, **kwargs))

            for param in block.parameters():
                param.requires_grad = False

    return get_trainable_parameters(model)
