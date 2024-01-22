import torch
import torch.nn as nn
import torch.nn.functional as F
from Simulator import schroedinger_simulator_2d
from ViscousFlowSolver import viscous_flow_solver_2d
from ConvNet import ConvNet
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
os.chdir(os.path.dirname(__file__))
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

grid_size = 256
xs = -torch.pi; xe = torch.pi
ys = -torch.pi; ye = torch.pi
dx = (xe-xs)/grid_size
dy = (ye-ys)/grid_size
hbar = 0.1
nu = 4e-3
dt = 0.01
Delta_t = 0.01
grid_t = 1001

schroedinger_solver = schroedinger_simulator_2d(
    device=device,
    n_grid=[grid_size, grid_size],
    range_x=[xs, xe],
    range_y=[ys, ye],
    hbar=hbar,
    dt=dt
)
schroedinger_solver.to(device)

viscous_solver = viscous_flow_solver_2d(
    n_grid=[grid_size, grid_size],
    range_x=[xs, xe],
    range_y=[ys, ye],
    nu=nu,
    dt=dt
)
viscous_solver.to(device)

# advectionNN = NNet(dt=dt, layers=[2,20,20,20,20,20,20,20,1], device=device)
advectionNN = ConvNet(dt)
advectionNN.to(device)

def display(psi1, psi2, x_np, y_np, i_step):
    ux, uy = schroedinger_solver.psi_to_velocity(psi1, psi2, hbar)
    wz = viscous_solver.vel_to_vor(ux, uy).to(torch.device('cpu'))
    w_np = torch.squeeze(wz).detach().cpu().numpy()
    vmax = 2
    vmin = -2
    levels = np.linspace(vmin, vmax, 20)
    cmap = mpl.cm.get_cmap('jet', 20)
    cs = plt.contourf(x_np,y_np,w_np,cmap=cmap,vmin=vmin,vmax=vmax,levels=levels)
    plt.pause(0.1)
    plt.savefig('results/'+str(i_step)+'.jpg')

def tube_vortex(mesh_x, mesh_y, type: int = 1):
    assert type > 0
    c1 = [-torch.pi/2, -torch.pi/2]
    c2 = [0, 0]
    rc = [torch.pi/3, torch.pi/6]
    psi1 = torch.ones(mesh_x.shape, dtype = torch.complex64)
    psi2 = 0.01 * torch.ones(mesh_x.shape, dtype = torch.complex64)
    for i in range(2):
        rx = (mesh_x-c1[i]/type)/rc[i]
        ry = (mesh_y-c2[i])/rc[i]
        r2 = rx**2+ry**2
        De = torch.exp(-(r2/9)**4)
        psi1 = psi1 * torch.complex(2*rx*De/(r2+1), (r2+1-2*De)/(r2+1))
    psi1 = psi1.to(device)
    psi2 = psi2.to(device)
    return psi1, psi2

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_type = 'train'):
        if data_type == 'train':
            pos_x = schroedinger_solver.mesh_x.to(torch.device('cpu'))
            pos_y = schroedinger_solver.mesh_y.to(torch.device('cpu'))
            psi1, psi2 = tube_vortex(pos_x, pos_y, 1)
            psi1 = psi1.to(device); psi2 = psi2.to(device)
            psi1, psi2 = schroedinger_solver.Normalization(psi1, psi2)
            psi1 = torch.fft.ifft2(torch.fft.fft2(psi1)*schroedinger_solver.kd)
            psi2 = torch.fft.ifft2(torch.fft.fft2(psi2)*schroedinger_solver.kd)
            for _ in range(10):
                psi1, psi2 = schroedinger_solver.Projection(psi1, psi2)
            psi1, psi2 = schroedinger_solver(psi1, psi2, 20*dt)
            # ux, uy = schroedinger_solver.psi_to_velocity(psi1, psi2, hbar)
            # ux_n, uy_n, _ = viscous_solver(ux, uy, Delta_t)
            psi1_n, psi2_n = schroedinger_solver(psi1, psi2, Delta_t)
            # ux_n, uy_n = schroedinger_solver.psi_to_velocity(psi1_n, psi2_n, hbar)
            psi_train = torch.cat((psi1.unsqueeze(0), psi2.unsqueeze(0)), 0).unsqueeze(0)
            psi_n_train = torch.cat((psi1_n.unsqueeze(0), psi2_n.unsqueeze(0)), 0).unsqueeze(0)

            for i_step in range(29):
                psi1 = psi1_n; psi2 =  psi2_n
                psi1_n, psi2_n = schroedinger_solver(psi1, psi2, Delta_t)
                # ux, uy = schroedinger_solver.psi_to_velocity(psi1, psi2, hbar)
                # ux_n, uy_n, _ = viscous_solver(ux, uy, Delta_t)
                # ux_n, uy_n = schroedinger_solver.psi_to_velocity(psi1_n, psi2_n, hbar)
                psi_tmp = torch.cat((psi1.unsqueeze(0), psi2.unsqueeze(0)), 0).unsqueeze(0)
                psi_n_tmp = torch.cat((psi1_n.unsqueeze(0), psi2_n.unsqueeze(0)), 0).unsqueeze(0)
                psi_train = torch.cat((psi_train, psi_tmp), 0)
                psi_n_train = torch.cat((psi_n_train, psi_n_tmp), 0)

            self.psi = psi_train
            self.psi_n = psi_n_train

    def __getitem__(self, index):
        return self.psi[index], self.psi_n[index]
    
    def __len__(self):
        return self.psi.shape[0]
    
def train():
    n_epoch = 2000
    optimizer = torch.optim.Adam(advectionNN.parameters(), lr=0.1, eps=1e-7)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    train_data_loader = torch.utils.data.DataLoader(Dataset('train'), batch_size=1)
    for epoch in range(n_epoch):
        advectionNN.train()
        total_loss = 0
        for batch_index, batch_data in enumerate(train_data_loader):
            psi, psi_n = batch_data
            psi.to(device)
            psi_n.to(device)

            psi_adv = advectionNN(psi)
            psi1_adv = psi_adv[0][0]; psi2_adv = psi_adv[0][1]
            psi1_adv, psi2_adv = schroedinger_solver.Normalization(psi1_adv, psi2_adv)
            psi1_adv, psi2_adv = schroedinger_solver.Projection(psi1_adv, psi2_adv)
            psi_adv = torch.cat((psi1_adv.unsqueeze(0), psi2_adv.unsqueeze(0)), 0).unsqueeze(0)
            # ux_pred, uy_pred = schroedinger_solver.psi_to_velocity(psi1_adv, psi2_adv, hbar)

            loss = F.l1_loss(psi_adv.real, psi_n.real) + F.l1_loss(psi_adv.imag, psi_n.imag)
            total_loss = total_loss + loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                print('Train Epoch: {} [{}]\t Loss: {:.10f}'.format(epoch, batch_index+1, loss))
            
        
        scheduler.step()
        if total_loss < 1e-5:
            break
        if epoch % 10 == 0:
            print('lr = ', optimizer.state_dict()['param_groups'][0]['lr'])
            # for name, param in advectionNN.named_parameters():
            #     print('-->name: ', name)
            #     print('-->para: ', param)
            #     print('-->grad: ', param.grad)
            #     break
    torch.save(advectionNN.state_dict(), 'model.pt')

def test():
    pos_x = schroedinger_solver.mesh_x.to(torch.device('cpu'))
    pos_y = schroedinger_solver.mesh_y.to(torch.device('cpu'))
    psi1, psi2 = tube_vortex(pos_x, pos_y, 1)
    psi1 = psi1.to(device); psi2 = psi2.to(device)
    psi1, psi2 = schroedinger_solver.Normalization(psi1, psi2)
    psi1 = torch.fft.ifft2(torch.fft.fft2(psi1)*schroedinger_solver.kd)
    psi2 = torch.fft.ifft2(torch.fft.fft2(psi2)*schroedinger_solver.kd)
    for _ in range(10):
        psi1, psi2 = schroedinger_solver.Projection(psi1, psi2)
    psi1, psi2 = schroedinger_solver(psi1, psi2, 20*dt)
    psi = torch.cat((psi1.unsqueeze(0), psi2.unsqueeze(0)), 0).unsqueeze(0)

    x_np = torch.squeeze(pos_x).detach().cpu().numpy()
    y_np = torch.squeeze(pos_y).detach().cpu().numpy()

    advectionNN.load_state_dict(torch.load('model.pt', map_location=lambda storage, location: storage))
    advectionNN.to(device)
    advectionNN.eval()
    with torch.no_grad():
        for i_step in range(grid_t):
            if i_step % 10 == 0:
                ux, uy = schroedinger_solver.psi_to_velocity(psi1, psi2, hbar)
                wz = viscous_solver.vel_to_vor(ux, uy).to(torch.device('cpu'))
                w_np = torch.squeeze(wz).detach().cpu().numpy()
                vmax = 2
                vmin = -2
                levels = np.linspace(vmin, vmax, 20)
                cmap = mpl.cm.get_cmap('jet', 20)
                cs = plt.contourf(x_np,y_np,w_np,cmap=cmap,vmin=vmin,vmax=vmax,levels=levels)
                plt.pause(0.1)
                s = '%03d'%int(i_step/10)
                plt.savefig('results/prediction/'+s+'.jpg')

            psi = advectionNN(psi)
            psi1 = psi[0][0]; psi2 = psi[0][1]
            psi1, psi2 = schroedinger_solver.Normalization(psi1, psi2)
            psi1, psi2 = schroedinger_solver.Projection(psi1, psi2)
            psi = torch.cat((psi1.unsqueeze(0), psi2.unsqueeze(0)), 0).unsqueeze(0)

if __name__ == '__main__':
    train()
    # test()