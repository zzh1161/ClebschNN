# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 23:22:40 2023

@author: shiying xiong
"""

import torch
import torch.utils.data
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

class FluidSolver(nn.Module):
    def __init__(self, grid_x, grid_y, hbar, dt):
        super(FluidSolver, self).__init__()
        xs = -torch.pi
        xe = torch.pi
        lx = xe - xs
        dx = lx / grid_x
        ys = -torch.pi
        ye = torch.pi
        ly = ye - ys
        dy = ly / grid_y
        x = torch.tensor(range(grid_x), dtype=torch.float32, requires_grad=False) * dx + xs
        y = torch.tensor(range(grid_y), dtype=torch.float32, requires_grad=False) * dy + ys
        (mesh_x,mesh_y) = torch.meshgrid(x,y)
        mesh_x = mesh_x.unsqueeze(0).unsqueeze(0)
        mesh_y = mesh_y.unsqueeze(0).unsqueeze(0)
        a = torch.tensor(range(grid_x), dtype=torch.float32, requires_grad=False)+grid_x/2
        b = torch.tensor(range(grid_y), dtype=torch.float32, requires_grad=False)+grid_y/2
        a = complex(0,1)*(a%grid_x-grid_x/2)
        b = complex(0,1)*(b%grid_y-grid_y/2)
        (kx,ky) = torch.meshgrid(a,b)
        k2 = kx*kx + ky*ky
        k2[0,0] = -1
        kx = kx.unsqueeze(0).unsqueeze(0)
        ky = ky.unsqueeze(0).unsqueeze(0)
        k2 = k2.unsqueeze(0).unsqueeze(0)
        self.mesh_x = nn.Parameter(mesh_x, requires_grad=False)
        self.mesh_y = nn.Parameter(mesh_y, requires_grad=False)
        
        kd0 = torch.zeros(1,1,grid_x,grid_y)
        kd1 = torch.ones(1,1,grid_x,grid_y)
        kd = torch.where(k2.real>-(grid_x**2+grid_y**2)/18., kd1, kd0)
        self.kd = nn.Parameter(kd, requires_grad=False)
        self.kx = nn.Parameter(kx, requires_grad=False)
        self.ky = nn.Parameter(ky, requires_grad=False)
        self.k2 = nn.Parameter(k2, requires_grad=False)
        self.kdx = nn.Parameter(kx*kd, requires_grad=False)
        self.kdy = nn.Parameter(ky*kd, requires_grad=False)
        self.kd2 = nn.Parameter(k2*kd, requires_grad=False)
        self.dt = dt
        self.hbar = hbar
        self.lx = lx
        self.ly = ly
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.SchroedingerMask = torch.exp(complex(0,0.5)*self.hbar*self.dt*self.k2)
        
    def Projection(self,psi1,psi2):
        div = torch.conj(torch.fft.ifft2(torch.fft.fft2(psi1)*self.k2))*complex(0,1)*psi1\
             +torch.conj(torch.fft.ifft2(torch.fft.fft2(psi2)*self.k2))*complex(0,1)*psi2
        div = torch.exp(complex(0,-1)*torch.fft.ifft2(torch.fft.fft2(div.real)/self.k2))
        return psi1*div, psi2*div
    
    def Normalization(self,psi1,psi2):
        psi_norm = torch.sqrt(torch.abs(psi1)**2 + torch.abs(psi2)**2)
        return psi1/psi_norm, psi2/psi_norm
    
    def Schroedinger(self,psi1,psi2):
        return torch.fft.ifft2(torch.fft.fft2(psi1)*self.SchroedingerMask), torch.fft.ifft2(torch.fft.fft2(psi2)*self.SchroedingerMask) 
    


    def forward(self,Delta_t,psi1,psi2):
        sub_steps = round(Delta_t/self.dt)
        for i_step in range(int(sub_steps)):
            [psi1,psi2] = self.Schroedinger(psi1,psi2)
            [psi1,psi2] = self.Normalization(psi1,psi2)
            [psi1,psi2] = self.Projection(psi1,psi2)
        return psi1, psi2


def to_np(x):
    return x.detach().cpu().numpy()

def plot_u(x, y, u, idx):
    fig = plt.figure()
    x = to_np(torch.squeeze(x))
    y = to_np(torch.squeeze(y))
    u = to_np(torch.squeeze(u))
    cs = plt.contourf(x,y,u)
    fig.colorbar(cs)
    plt.show()
    plt.savefig('results/'+str(idx)+'.jpg')
    


def tube_vortex(mesh_x,mesh_y):
    c1 = [-torch.pi/2, -torch.pi/2]
    c2 = [0, 0]
    rc = [torch.pi/3, torch.pi/6]
    psi1 = torch.ones(mesh_x.shape, dtype = torch.complex64)
    psi2 = 0.01 * torch.ones(mesh_x.shape, dtype = torch.complex64)
    psi1 = psi1.to(device)
    psi2 = psi2.to(device)
    for i in range(2):
        rx = (mesh_x-c1[i])/rc[i]
        ry = (mesh_y-c2[i])/rc[i]
        r2 = rx**2+ry**2
        De = torch.exp(-(r2/9)**4)
        psi1 = psi1 * torch.complex(2*rx*De/(r2+1), (r2+1-2*De)/(r2+1))
    return psi1, psi2

def psi_to_vel(psi1,psi2,kx,ky,hbar):
    psi = torch.fft.fft2(psi1)
    psix = torch.fft.ifft2(psi*kx)
    psiy = torch.fft.ifft2(psi*ky)
    ux = torch.conj(psix)*complex(0,1)*psi1
    uy = torch.conj(psiy)*complex(0,1)*psi1
    psi = torch.fft.fft2(psi2)
    psix = torch.fft.ifft2(psi*kx)
    psiy = torch.fft.ifft2(psi*ky)
    ux = ux + torch.conj(psix)*complex(0,1)*psi2
    uy = uy + torch.conj(psiy)*complex(0,1)*psi2
    return ux.real*hbar, uy.real*hbar
    
def vor_to_vel(wz,kx,ky,k2): 
    return - torch.fft.ifft2(torch.fft.fft2(wz)*ky/k2).real, torch.fft.ifft2(torch.fft.fft2(wz)*kx/k2).real


def vel_to_vor(ux,uy,kx,ky):
    return torch.fft.ifft2(kx*torch.fft.fft2(uy)-ky*torch.fft.fft2(ux)).real


dt = 0.03
Delta_t = 0.3
grid_x = 512
grid_y = 512
grid_t = 101
hbar = 0.1


fig = plt.figure()

solver = FluidSolver(grid_x, grid_y, hbar, dt)
solver.to(device)

x_np = to_np(torch.squeeze(solver.mesh_x))
y_np = to_np(torch.squeeze(solver.mesh_y))

[psi1,psi2] = tube_vortex(solver.mesh_x,solver.mesh_y)
psi1 = psi1.to(device)
psi2 = psi2.to(device)
[psi1,psi2] = solver.Normalization(psi1,psi2)
psi1 = torch.fft.ifft2(torch.fft.fft2(psi1)*solver.kd)
psi2 = torch.fft.ifft2(torch.fft.fft2(psi2)*solver.kd)


for i_step in range(10):
    [psi1,psi2] = solver.Projection(psi1,psi2)
    

    
for i_step in range(grid_t):
    [ux,uy] = psi_to_vel(psi1,psi2,solver.kdx,solver.kdy,hbar)
    [wz] = vel_to_vor(ux,uy,solver.kdx,solver.kdy)
    w_np = to_np(torch.squeeze(wz))
    vmax = 2
    vmin = -2
    levels = np.linspace(vmin, vmax, 20)
    cmap = mpl.cm.get_cmap('jet', 20)
    cs = plt.contourf(x_np,y_np,w_np,cmap=cmap,vmin=vmin,vmax=vmax,levels=levels)
    plt.pause(0.1)
    plt.savefig('results/'+str(i_step)+'.jpg')
    [psi1,psi2] = solver(Delta_t,psi1,psi2)