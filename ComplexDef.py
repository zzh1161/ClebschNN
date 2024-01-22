import torch
import torch.nn as nn
import torch.nn.functional as F

class complexReLU(nn.Module):
    def __init__(self, inplace: bool = False):
        super(complexReLU, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        return F.relu(input.real, self.inplace).type(torch.complex64) + 1j*F.relu(input.imag, self.inplace).type(torch.complex64)
    
class complexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding = 0,
                 padding_mode='zeros', dilation=1, groups=1, bias=True):
        super(complexConv2d, self).__init__()
        self.conv_r = nn.Conv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            padding_mode = padding_mode
        )
        self.conv_i = nn.Conv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            padding_mode = padding_mode
        )
        
    def forward(self, input, dtype=torch.complex64):    
        return (self.conv_r(input.real) - self.conv_i(input.imag)).type(dtype) +\
               1j*(self.conv_r(input.imag) + self.conv_i(input.real)).type(dtype)
    
def complex_clamp(x, min, max):
    real_p = torch.clamp(x.real, 0.5*min, 0.5*max)
    imag_p = torch.clamp(x.imag, 0.5*min, 0.5*max)
    return real_p + 1j*imag_p