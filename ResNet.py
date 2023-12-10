from typing import Any, Callable, List, Optional, Type, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from ComplexDef import complexReLU, complexConv2d
import complexPyTorch.complexLayers as cptl
import complexPyTorch.complexFunctions as cptf

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
            norm_layer = cptl.NaiveComplexBatchNorm2d
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
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

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
            norm_layer = cptl.NaiveComplexBatchNorm2d
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
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

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
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = cptl.NaiveComplexBatchNorm2d
        self.norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = complexConv2d(
            in_channels  = 2,
            out_channels = self.inplanes,
            kernel_size  = 7,
            stride       = 1,
            padding      = 3,
            padding_mode = 'circular',
            bias         = False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = complexReLU(inplace = False)
        self.maxpool = cptl.ComplexMaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = cptl.ComplexAvgPool2d((1,1))
        self.fc = conv1x1(512, 1)

        for m in self.modules():
            if isinstance(m, complexConv2d):
                nn.init.kaiming_normal_(m.conv_r.weight, mode='fan_out', nonlinearity='relu')
                nn.init.kaiming_normal_(m.conv_i.weight, mode='fan_out', nonlinearity='relu')
            
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    if m.bn2.bn_r.weight is not None:
                        nn.init.constant_(m.bn2.bn_r.weight, 0)
                    if m.bn2.bn_i.weight is not None:
                        nn.init.constant_(m.bn2.bn_i.weight, 0)
                elif isinstance(m, Bottleneck):
                    if m.bn3.bn_r.weight is not None:
                        nn.init.constant_(m.bn3.bn_r.weight, 0)
                    if m.bn3.bn_i.weight is not None:
                        nn.init.constant_(m.bn3.bn_i.weight, 0)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential():
        norm_layer = self.norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes*block.expansion, stride),
                norm_layer(planes*block.expansion)
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups     = self.groups,
                    base_width = self.base_width,
                    dilation   = self.dilation,
                    norm_layer = norm_layer
                )
            )
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = self.fc(out)

        return out
    
class SchrodingerResNet(nn.Module):
    def __init__(
        self,
        hbar: float,
        dx: float,
        dt: float,
        k2: torch.Tensor,
        layers: List[int] = [2,2,1,1],
        zero_init: bool = True,
        device: torch.device = torch.device('cpu')
    ):
        self.hbar = hbar
        self.dx   = dx
        self.dt   = dt
        self.k2   = k2.to(device)
        self.SchroedingerMask = torch.exp(complex(0,0.5)*self.hbar*self.dt*self.k2)

        self.resnet = ResNet(BasicBlock, layers, zero_init)

    def forward(self, psi):
        f = self.resnet(psi)
        psi1 = psi[:,0:1,:,:]
        psi2 = psi[:,1:2,:,:]

        psi1 = torch.fft.ifft2(torch.fft.fft2(psi1, dim=(-2,-1))*self.SchroedingerMask, dim=(-2,-1))
        psi2 = torch.fft.ifft2(torch.fft.fft2(psi2, dim=(-2,-1))*self.SchroedingerMask, dim=(-2,-1))
        psi1 = torch.sqrt(1 - self.dt**2 * torch.abs(f)**2)*psi1 - f*torch.conj(psi2)*self.dt
        psi2 = torch.sqrt(1 - self.dt**2 * torch.abs(f)**2)*psi2 + f*torch.conj(psi1)*self.dt

        # print('f.mean = ', torch.mean(torch.abs(f)))
        return torch.cat((psi1, psi2), dim=1)