import torch

from modules.normalization import Conv2dWS
from torch import nn

class ConvAdapter(nn.Module):
    def __init__(self, inplanes, outplanes, width, 
                kernel_size=1, padding=1, stride=1, act_layer=nn.ReLU, **kwargs):
        super().__init__()

        # Depth-wise conv
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=kernel_size, stride=stride, groups=width, padding=padding, bias=False)
        self.act = act_layer()
        # Point-wise conv
        self.conv2 = nn.Conv2d(width, outplanes, kernel_size=1, stride=1, bias=False)
        self.se = nn.Parameter(torch.zeros((1, outplanes, 1, 1)), requires_grad=True)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        out = out * self.se

        return out
    
class TestAdapter(nn.Module):
    def __init__(self, inplanes, outplanes, width, 
                kernel_size=1, padding=1, stride=1, act_layer=nn.ReLU, weight_standardization=False, **kwargs):
        super().__init__()

        # Downsample
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride, padding=padding, groups=int(outplanes//width) ,bias=False)
        self.act = act_layer()
        # Regular conv
        self.conv2 = nn.Conv2d(outplanes, outplanes, kernel_size=1, bias=False)
        self.se = nn.Parameter(torch.zeros((1, outplanes, 1, 1)), requires_grad=True)

        if weight_standardization:
            self.conv1 = Conv2dWS(self.conv1)
            self.conv2 = Conv2dWS(self.conv2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        out = out * self.se

        return out
    
class SequentialAdapter(nn.Module):
    def __init__(self, module, module_cls, *args, **kwargs):
        super(ParallelAdapter, self).__init__()
        self.original_module = module
        self.sequential_module = module_cls(*args, **kwargs)

    def forward(self, x):
        x = self.original_module(x)

        return x + self.sequential_module(x) 

class ParallelAdapter(nn.Module):
    def __init__(self, module, module_cls, *args, **kwargs):
        super(ParallelAdapter, self).__init__()
        self.original_module = module
        self.parallel_module = module_cls(*args, **kwargs)

    def forward(self, x):
        return self.original_module(x) + self.parallel_module(x)
