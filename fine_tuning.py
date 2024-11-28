import static
import torch

from peft import get_peft_model, LoraConfig

def unfreeze_peft_model(model):
    for name, module in model.named_modules():
        if name.endswith('.base_layer'): 
            continue # If not original layer before lora

        for param_name, param in module.named_parameters(): 
            if '.' in param_name:
                continue # Only parameters of self

            param.requires_grad = True

def get_lora_model(model, target_modules, r=static.LORA_RANK, lora_alpha=static.LORA_ALPHA, lora_dropout=static.LORA_DROPOUT):

    model = get_peft_model(model, LoraConfig(target_modules=target_modules, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout))
    unfreeze_peft_model(model)
    model.to(static.DEVICE)

    return model

def get_pruning_mask(param, amount):
    with torch.no_grad():
        # Flatten the weights and find the magnitude thresholds
        flattened_weights = param.abs().view(-1)
        threshold_index = int(len(flattened_weights) * amount)
        # Sort to find the threshold value
        threshold_value = torch.sort(flattened_weights)[0][threshold_index]
        # Create a mask: True for weights above the threshold, False otherwise
        return param.abs() >= threshold_value

def prune_gradients(module, name, amount):
    param = getattr(module, name)
    mask = get_pruning_mask(param, amount=amount)
    param.register_hook(lambda grad: grad * mask)