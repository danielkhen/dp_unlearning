import torch

from torch import nn

NORM_LAYERS = (nn.GroupNorm, nn.LayerNorm, nn.BatchNorm2d)

class ConvAdapter(nn.Module):
    def __init__(self, inplanes, outplanes, width, 
                kernel_size=3, padding=1, stride=1, dilation=1, norm_layer=None, act_layer=None, weight_standardization=False, **kwargs):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.Identity
        if act_layer is None:
            act_layer = nn.Identity

        # Depth-wise conv
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=kernel_size, stride=stride, groups=width, padding=padding, dilation=int(dilation), bias=False)
        nn.init.zeros_(self.conv1.weight)
        self.norm1 = norm_layer(width)
        self.act = act_layer()
        # Point-wise conv
        self.conv2 = nn.Conv2d(width, outplanes, kernel_size=1, stride=1, bias=False)
        nn.init.zeros_(self.conv2.weight)
        self.norm2 = norm_layer(outplanes)
        self.se = nn.Parameter(1.0 * torch.ones((1, outplanes, 1, 1)), requires_grad=True)

        if weight_standardization:
            self.conv1=Conv2dWS(self.conv1)
            self.conv2=Conv2dWS(self.conv2)

    
    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = out * self.se

        return out
    
class ParallelBlockAdapter(nn.Module):
    def __init__(self, block, bottleneck_ratio=4, weight_standardization=False):
        super(ParallelBlockAdapter, self).__init__()
        self.residual_block = block
        self.bottleneck_ratio = bottleneck_ratio
        conv = block.conv1
        self.adapter = ConvAdapter(conv.in_channels, conv.out_channels, width=conv.in_channels // bottleneck_ratio,
                                   kernel_size=conv.kernel_size, stride=conv.stride, padding=conv.padding, weight_standardization=weight_standardization, act_layer=nn.ReLU)
    
    def forward(self, x):
        residual = self.residual_block(x)
        adapter_output = self.adapter(x)

        return residual + adapter_output
    
class SequentialBlockAdapter(nn.Module):
    def __init__(self, block, bottleneck_ratio=4, weight_standardization=False):
        super(SequentialBlockAdapter, self).__init__()
        self.residual_block = block
        self.bottleneck_ratio = bottleneck_ratio
        conv = block.conv2
        self.adapter = ConvAdapter(conv.out_channels, conv.out_channels, width=conv.out_channels // bottleneck_ratio,
                                   kernel_size=conv.kernel_size, padding=conv.padding, weight_standardization=weight_standardization, act_layer=nn.ReLU)
    
    def forward(self, x):
        residual = self.residual_block(x)
        adapter_output = self.adapter(residual)
        
        return residual + adapter_output

class Conv2dWS(nn.Conv2d):
    def __init__(self, original_conv):
        # Initialize with the same attributes as the original nn.Conv2d
        super(Conv2dWS, self).__init__(
            in_channels=original_conv.in_channels,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            dilation=original_conv.dilation,
            groups=original_conv.groups,
            bias=original_conv.bias is not None
        )

        # Copy over the weights and biases
        self.weight.data = original_conv.weight.data.clone()
        
        if original_conv.bias is not None:
            self.bias.data = original_conv.bias.data.clone()

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=(1, 2, 3), keepdim=True)
        std = weight.std(dim=(1, 2, 3), keepdim=True)
        weight = (weight - weight_mean) / std

        return nn.functional.conv2d(x, weight, self.bias, self.stride,
                                self.padding, self.dilation, self.groups)

# Implementation with parameterization didn't work with grad_sample_mode=hooks and didn't have performance benefits
def standardize_model(model):
    children = list(model.named_children())

    for idx, (name, child) in enumerate(children):
        next_child = children[idx + 1][1] if idx + 1 < len(children) else None
        
        if isinstance(child, nn.Conv2d) and isinstance(next_child, NORM_LAYERS):
            setattr(model, name, Conv2dWS(child))
        else:
            standardize_model(child) # Continue recursivly for child modules