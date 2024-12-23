import torch
import torch.nn.functional as F

from torch import nn

class KernelExpand(nn.Module):
    def __init__(self, kernel_size, stride, padding):
        super(KernelExpand, self).__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        if isinstance(stride, int):
            stride = (stride, stride)

        if isinstance(padding, int):
            padding = (padding, padding, padding, padding)
        else:
            padding = (padding[0], padding[0], padding[1], padding[1])

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        height = x.size(2)
        width = x.size(3)

        padded = F.pad(x, self.padding)
        transforms = []
        
        for kernel_height in self.kernel_size[0]:
            for kernel_width in self.kernel_size[1]:
                transforms.append(padded[:, :, kernel_height:kernel_height + height:self.stride[0], kernel_width:kernel_width + width:self.stride[1]])

        return torch.cat(transforms, dim=1)

class KernelSum(nn.Module):
    def __init__(self, kernel_size):
        super(KernelSum, self).__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.kernel_mult = kernel_size[0] * kernel_size[1]

def kernel_sum(self, x):
    reshape_size = (x.size(0), self.kernel_mult, x.size(1) // self.kernel_mult, x.size(2), x.size(3))
    reshaped = x.reshape(reshape_size)
    summed = reshaped.sum(dim=1)

    return summed

class KernelConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, groups=1, bias=False):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        kernel_mult = kernel_size[0] * kernel_size[1]
        super(KernelConv2d, self).__init__(in_channels * kernel_mult, out_channels * kernel_mult, kernel_size=1, groups=groups * kernel_mult, bias=bias)