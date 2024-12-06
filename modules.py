from torch.nn import Module, Conv2d, GroupNorm, LayerNorm, BatchNorm2d
import torch.nn.utils.parametrize as parametrize

NORM_LAYERS = (GroupNorm, LayerNorm, BatchNorm2d)

# Weight standardization parameterization module
class Standardization(Module):
    def forward(self, X):
        mean = X.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        std = X.view(X.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5

        return (X - mean) / std.expand_as(X)
    
def standardize_model(model):
    modules = list(model.modules())
    
    # Register weight standardization parameterization if layer is Conv2d and next layer is a normalization layer
    for module, next_module in zip(modules[:-1], modules[1:]):
        if isinstance(module, Conv2d) and isinstance(next_module, NORM_LAYERS):
            parametrize.register_parametrization(module, "weight", Standardization())