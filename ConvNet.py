import torch
import torch.nn as nn
import torch.nn.functional as F
from ComplexDef import complexReLU, complexConv2d
import complexPyTorch.complexLayers as cptl
import complexPyTorch.complexFunctions as cptf

def convnxn(in_channels: int, out_channels: int, kernel_size: int,
            stride: int = 1, groups: int = 1, dilation: int = 1):
    """nxn complex convolution with circular padding"""
    return complexConv2d(
        in_channels  = in_channels,
        out_channels = out_channels,
        kernel_size  = kernel_size,
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

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        
        self.conv1 = convnxn(in_channels, out_channels, kernel_size)
        self.conv2 = convnxn(in_channels, out_channels, kernel_size)
        if kernel_size == 1:
            self.conv1 = conv1x1(in_channels, out_channels)
            self.conv2 = conv1x1(in_channels, out_channels)
        self.relu = complexReLU(inplace=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = cptf.complex_relu(out)
        return out
    
class ConvNet(nn.Module):
    def __init__(
        self,
        dt: float
    ):
        super().__init__()
        self.dt   = dt
        self.nn   = nn.Sequential(
            ConvBlock(2, 16),
            ConvBlock(16, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 128, kernel_size=1),
            conv1x1(128, 1)
        )

    def forward(self, psi):
        f = self.nn(psi)
        psi1 = psi[:,0:1,:,:]
        psi2 = psi[:,1:2,:,:]

        psi1 = torch.sqrt(1 - self.dt**2 * torch.abs(f)**2)*psi1 - f*torch.conj(psi2)*self.dt
        psi2 = torch.sqrt(1 - self.dt**2 * torch.abs(f)**2)*psi2 + f*torch.conj(psi1)*self.dt

        # print('f.mean = ', torch.mean(torch.abs(f)))
        return torch.cat((psi1, psi2), dim=1)