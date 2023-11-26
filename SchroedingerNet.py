import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from complexPyTorch.complexLayers import ComplexConv2d
from complexPyTorch.complexFunctions import complex_relu

class complexReLU(nn.Module):
    def __init__(self, inplace: bool = False):
        super(complexReLU, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        return F.relu(input.real, self.inplace).type(torch.complex64) + 1j*F.relu(input.imag, self.inplace).type(torch.complex64)

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size=3):
        super(ResidualBlock, self).__init__()
        self.kernel_size = kernel_size
        self.left = nn.Sequential(
            ComplexConv2d(inchannel, outchannel, kernel_size=self.kernel_size, padding=(kernel_size-1)//2),
            complexReLU(inplace=False),
            ComplexConv2d(outchannel, outchannel, kernel_size=1, padding=0),
        )
        self.shortcut = nn.Sequential()
        if inchannel != outchannel:
            self.shortcut = nn.Sequential(
                ComplexConv2d(inchannel, outchannel, kernel_size=1, padding=0),
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = complex_relu(out)
        return out
    
class schroedinger_net(nn.Module):
    def __init__(self, hbar, dx, dt, k2, device=torch.device('cpu')):
        super(schroedinger_net, self).__init__()
        self.conp = 2 # components of psi, i.e. psi1 and psi2
        # self.hide = 8 
        self.hbar = hbar
        self.dx = dx
        self.dt = dt
        self.k2 = k2.to(device)

        self.nn = nn.Sequential(
            ResidualBlock(self.conp, 16),
            ResidualBlock(16, 16),
            ResidualBlock(16, 32),
            ResidualBlock(32, 64),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64, kernel_size=1),
            ComplexConv2d(64, 1, kernel_size=1, padding=0),
        )

    def forward(self, psi0):
        f = self.nn(psi0)
        psi1 = psi0[:,0:1,:,:]
        psi2 = psi0[:,1:2,:,:]
        dpsi1dt = -f*torch.conj(psi2) + 0.5j*self.hbar*torch.fft.ifftn(torch.fft.fftn(psi1, dim=(-2,-1))*self.k2, dim=(-2,-1))
        dpis2dt = f*torch.conj(psi1)  + 0.5j*self.hbar*torch.fft.ifftn(torch.fft.fftn(psi2, dim=(-2,-1))*self.k2, dim=(-2,-1))
        psi1 = psi1 + dpsi1dt*self.dt
        psi2 = psi2 + dpis2dt*self.dt
        return torch.cat((psi1, psi2), dim=1)