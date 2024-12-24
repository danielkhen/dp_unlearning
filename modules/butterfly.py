from torch import nn
from modules.convolutions import KernelExpand, KernelSum, KernelConv2d

def count_divisions(base, number):
        divisions = 0

        while number % base == 0:
            number //= base
            divisions += 1

        return divisions, number

class ButterflyPermutation(nn.Module):
    """
    A permutation that helps flow each in channel to each out channel in a width x width butterfly operation.
    Each channel is assumed to be connected to a group given by group_size, the permutation spreads the channels
    to group_size new groups of size multiplier (which will then be convoluted to create the actual connections).
    After the convolution we expect a group size of group_size x multiplier that is why each "batch" of multiplier groups
    or group_size x multiplier channels is permutated with the same batch on the output.
    The permutation is created in the following way:
    1. The index in the group of the permutation is the index of the group in the batch of the input.
    2. The index of the group in the batch of the permutation is the index in the group of the input.
    3. The index of the batch of the permutation is the index of the batch of the input.
    """
    def __init__(self, width, group_size, multiplier, kernel_size=1):
        super(ButterflyPermutation, self).__init__()
        batch_size = group_size * multiplier # Group of multiplier x groups
        self.permutation = []

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        kernel_mult = kernel_size[0] * kernel_size[1]

        for idx in range(width * kernel_mult):
            idx_in_group = idx % group_size
            group_idx = (idx % batch_size) // group_size
            batch_idx = (idx % width) // batch_size
            kernel_idx = idx // width
            self.permutation.append(group_idx + multiplier * idx_in_group + batch_size * batch_idx + kernel_idx * width)
    
    def forward(self, x):
        return x[:, self.permutation, :, :]

class ButterflyConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, base=4, width=None, bias=False):
        super(ButterflyConv2d, self).__init__()

        if not width:
            width = min(in_channels, out_channels)
        
        assert in_channels % width == 0
        assert out_channels % width == 0

        self.bias = bias
        self.kernel_size = kernel_size
        
        contraction_count, contraction_remainder = count_divisions(base, in_channels // width)
        butterfly_count, butterfly_remainder = count_divisions(base, width)
        expansion_count, expansion_remainder = count_divisions(base, out_channels // width)
        orig_butterfly_remainder = butterfly_remainder

        if self.kernel_size != 1:
            self.kernel_expand = KernelExpand(kernel_size, stride, padding)

        self.contractions = nn.Sequential()
        self.butterflies = nn.Sequential()
        self.expansions = nn.Sequential()

        if self.kernel_size != 1:
            self.kernel_sum = KernelSum(kernel_size)

        for idx in range(contraction_count):
            in_channels_ = in_channels // (base ** idx)
            self.add_contraction(in_channels_, base)

        if contraction_remainder != 1:
            in_channels_ = width * contraction_remainder
            self.add_contraction(in_channels_, contraction_remainder, group_size=butterfly_remainder)

            if butterfly_remainder != 1:
                self.add_butterfly_permutation(width, base ** butterfly_count, orig_butterfly_remainder)
                butterfly_remainder = 1

        for idx in range(butterfly_count):
            self.add_butterfly(width, base)

            if idx != butterfly_count - 1:
                self.add_butterfly_permutation(width, base ** (idx + 1), base)

        if expansion_remainder != 1:
            self.add_expansion(width, expansion_remainder, group_size=butterfly_remainder)

        if butterfly_remainder != 1:
            self.add_butterfly_permutation(width, base ** butterfly_count, orig_butterfly_remainder)

            if expansion_remainder == 1:
                self.add_butterfly(width, butterfly_remainder)

        for idx in range(expansion_count, 0, -1):
            in_channels_ = out_channels // (base ** idx)
            self.add_expansion(in_channels_, base)

    def add_contraction(self, in_channels, multiplier, group_size=1):
        out_channels = in_channels // multiplier
        name = f'cont_in{in_channels}_out{out_channels}_g{group_size}_m{multiplier}'
        self.contractions.add_module(name, KernelConv2d(in_channels, out_channels, self.kernel_size, groups=out_channels // group_size, bias=self.bias))

    def add_expansion(self, in_channels, multiplier, group_size=1):
        out_channels = in_channels * multiplier
        name = f'exp_in{in_channels}_out{out_channels}_g{group_size}_m{multiplier}'
        self.expansions.add_module(name, KernelConv2d(in_channels, out_channels, self.kernel_size, groups=in_channels // group_size, bias=self.bias))

    def add_butterfly(self, width, group_size):
        idx = len(self.butterflies) // 2
        name = f'bf_i{idx}_w{width}_g{group_size}'
        self.butterflies.add_module(name, KernelConv2d(width, width, self.kernel_size, groups=width // group_size, bias=self.bias))

    def add_butterfly_permutation(self, width, group_size, multiplier):
        name = f'bfperm_w{width}_g{group_size}_m{multiplier}'
        self.butterflies.add_module(name, ButterflyPermutation(width, group_size, multiplier, self.kernel_size))

    def forward(self, x):
        if self.kernel_size != 1:
            x = self.kernel_expand(x)

        x = self.contractions(x)
        x = self.butterflies(x)
        x = self.expansions(x)

        if self.kernel_size != 1:
            x = self.kernel_sum(x)

        return x