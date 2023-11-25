import torch
import torch.nn as nn
import torch.nn.functional as F
from Simulator import schroedinger_simulator_2d
from SchroedingerNet import schroedinger_net
from ViscousFlowSolver import viscous_flow_solver_2d
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
os.chdir(os.path.dirname(__file__))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
grid_size = 256
xs = -torch.pi; xe = torch.pi
ys = -torch.pi; ye = torch.pi
dx = (xe-xs)/grid_size
dy = (ye-ys)/grid_size
hbar = 0.1
nu = 4e-4
dt = 0.03
Delta_t = 0.03
grid_t = 100

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

advectionNN = schroedinger_net(dx=dx, eps=dt)
advectionNN.to(device)

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
            ux, uy = schroedinger_solver.psi_to_velocity(psi1, psi2, hbar)
            ux_n, uy_n, _ = viscous_solver(ux, uy, Delta_t)
            psi_train_1 = torch.cat((psi1.unsqueeze(0),psi2.unsqueeze(0)), 0).unsqueeze(0)
            vel_next_1 = torch.cat((ux_n.unsqueeze(0), uy_n.unsqueeze(0)), 0).unsqueeze(0)

            psi1, psi2 = tube_vortex(pos_x, pos_y, 2)
            ux, uy = schroedinger_solver.psi_to_velocity(psi1, psi2, hbar)
            ux_n, uy_n, _ = viscous_solver(ux, uy, Delta_t)
            psi_train_2 = torch.cat((psi1.unsqueeze(0),psi2.unsqueeze(0)), 0).unsqueeze(0)
            vel_next_2 = torch.cat((ux_n.unsqueeze(0), uy_n.unsqueeze(0)), 0).unsqueeze(0)

            psi1, psi2 = tube_vortex(pos_x, pos_y, 3)
            ux, uy = schroedinger_solver.psi_to_velocity(psi1, psi2, hbar)
            ux_n, uy_n, _ = viscous_solver(ux, uy, Delta_t)
            psi_train_3 = torch.cat((psi1.unsqueeze(0),psi2.unsqueeze(0)), 0).unsqueeze(0)
            vel_next_3 = torch.cat((ux_n.unsqueeze(0), uy_n.unsqueeze(0)), 0).unsqueeze(0)

            self.psi = torch.cat((psi_train_1, psi_train_2, psi_train_3), 0)
            self.vel = torch.cat((vel_next_1, vel_next_2, vel_next_3), 0)
    
    def __getitem__(self, index):
        return self.psi[index], self.vel[index]
    
    def __len__(self):
        return self.psi.shape[0]
    
def train():
    n_epoch = 100
    optimizer = torch.optim.Adam(advectionNN.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    train_data_loader = torch.utils.data.DataLoader(Dataset('train'), batch_size=1)
    for epoch in range(n_epoch):
        advectionNN.train()
        for batch_index, batch_data in enumerate(train_data_loader):
            psi, vel = batch_data
            psi.to(device)
            vel.to(device)

            ux_ground = vel[0][0]; uy_ground = vel[0][1]

            psi_adv = advectionNN(psi, Delta_t)
            psi1_adv = psi_adv[0][0]; psi2_adv = psi_adv[0][1]
            psi1_adv, psi2_adv = schroedinger_solver.Normalization(psi1_adv, psi2_adv)
            psi1_adv, psi2_adv = schroedinger_solver.Projection(psi1_adv, psi2_adv)
            ux_pred, uy_pred = schroedinger_solver.psi_to_velocity(psi1_adv, psi2_adv, hbar)

            loss = F.mse_loss(ux_pred, ux_ground) + F.mse_loss(uy_pred, uy_ground)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if epoch % 1 == 0:
                # print('ux = ', ux)
                # print('uy = ', uy)
                # print('ux_pred = ', ux_pred)
                # print('uy_pred = ', uy_pred)
                # print('ux_ground = ', ux_ground)
                # print('uy_ground = ', uy_ground)
                print('Train Epoch: {} [{}/3]\t Loss: {:.6f}'.format(epoch, batch_index+1, loss))
        scheduler.step()
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
    psi = torch.cat((psi1.unsqueeze(0), psi2.unsqueeze(0)), 0).unsqueeze(0)

    x_np = torch.squeeze(pos_x).detach().cpu().numpy()
    y_np = torch.squeeze(pos_y).detach().cpu().numpy()
    fig = plt.figure()

    advectionNN.load_state_dict(torch.load('model.pt', map_location=lambda storage, location: storage))
    advectionNN.to(device)
    advectionNN.eval()
    with torch.no_grad():
        for i_step in range(grid_t):
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

            psi = advectionNN(psi, Delta_t)
            psi1 = psi[0][0]; psi2 = psi[0][1]
            psi1, psi2 = schroedinger_solver.Normalization(psi1, psi2)
            psi1, psi2 = schroedinger_solver.Projection(psi1, psi2)
            psi = torch.cat((psi1.unsqueeze(0), psi2.unsqueeze(0)), 0).unsqueeze(0)

if __name__ == '__main__':
    # train()
    test()