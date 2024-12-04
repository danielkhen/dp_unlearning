from torch.nn import Conv2d, GroupNorm, LayerNorm, BatchNorm2d, functional

NORM_LAYERS = (GroupNorm, LayerNorm, BatchNorm2d)

class Conv2dWS(Conv2d):
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
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)

        return functional.conv2d(x, weight, self.bias, self.stride,
                                self.padding, self.dilation, self.groups)
    
    @staticmethod
    def replace_conv2d_with_ws(model):
        children = list(model.named_children())
        
        for idx, (name, child) in enumerate(children):
            next_child = children[idx + 1][1] if idx + 1 < len(children) else None
            
            if isinstance(child, (Conv2d)) and isinstance(next_child, NORM_LAYERS):
                setattr(model, name, Conv2dWS(child))
            else:
                Conv2dWS.replace_conv2d_with_ws(child) # Continue recursivly for child modules