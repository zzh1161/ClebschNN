from typing import Any, Callable, List, Optional, Type, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

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
        # nn.init.kaiming_normal_(self.conv_r.weight, mode='fan_out', nonlinearity='relu')
        # nn.init.kaiming_normal_(self.conv_i.weight, mode='fan_out', nonlinearity='relu')
        
    def forward(self, input, dtype=torch.complex64):    
        return (self.conv_r(input.real) - self.conv_i(input.imag)).type(dtype) +\
               1j*(self.conv_r(input.imag) + self.conv_i(input.real)).type(dtype)

def conv3x3(in_channels: int, out_channels: int, stride: int = 1, groups: int = 1, dilation: int = 1):
    """3x3 complex convolution with circular padding"""
    return complexConv2d(
        in_channels  = in_channels,
        out_channels = out_channels,
        kernel_size  = 3,
        stride       = stride,
        padding      = dilation,
        padding_mode = 'circular',
        dilation     = dilation,
        groups       = groups,
        bias         = False
    )

def conv1x1(in_channels: int, out_channels: int, stride: int = 1):
    """1x1 complex convolution"""
    return complexConv2d(
        in_channels  = in_channels,
        out_channels = out_channels,
        kernel_size  = 1,
        stride       = stride,
        bias         = False
    )
    
class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation>1 is not supported in BasicBlock")
        self.conv1      = conv3x3(in_channels, out_channels, stride)
        self.bn1        = norm_layer(out_channels)
        self.relu       = complexReLU(inplace = False)
        self.conv2      = conv3x3(out_channels, out_channels)
        self.bn2        = norm_layer(out_channels)
        self.downsample = downsample
        self.stride     = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        # out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(out_channels * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(in_channels, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, out_channels * self.expansion)
        self.bn3 = norm_layer(out_channels * self.expansion)
        self.relu = complexReLU(inplace = False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        # out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        # out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        zero_init_residual: bool = False,
        groups: int = 1,
        
    )