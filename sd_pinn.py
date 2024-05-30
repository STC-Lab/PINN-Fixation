
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from scipy.io import loadmat
import dataprocessing
import time
from pyDOE import lhs



class FCN(nn.Module):
    "Defines a standard fully-connected network in PyTorch"
    
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(*[
                        nn.Linear(N_INPUT, N_HIDDEN),
                        activation()])
        self.fch = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(N_HIDDEN, N_HIDDEN),
                            activation()]) for _ in range(N_LAYERS-1)])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)

    def forward(self, input):
        x = torch.cat(input, dim=-1)
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x
    

class FNet(nn.Module):
    "Defines a standard fully-connected network in PyTorch"
    
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(*[
                        nn.Linear(N_INPUT, N_HIDDEN),
                        activation()])
        self.fch = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(N_HIDDEN, N_HIDDEN),
                            activation()]) for _ in range(N_LAYERS-1)])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)

    def forward(self, input):
        #x = torch.cat(input, dim=-1)
        x = input
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x

    
class GNet(nn.Module):
    "Defines a standard fully-connected network in PyTorch"
    
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(*[
                        nn.Linear(N_INPUT, N_HIDDEN),
                        activation()])
        self.fch = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(N_HIDDEN, N_HIDDEN),
                            activation()]) for _ in range(N_LAYERS-1)])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)

    def forward(self, input):
        #x = torch.cat(input, dim=-1)
        x = input
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x

torch.set_default_dtype(torch.float)
torch.manual_seed(1234)
np.random.seed(1231)
# first, create some noisy observational data
file = './dataset/sd_pinn_inverse.mat'
x,t,u = dataprocessing.load_data(file)
X,T,U = dataprocessing.totensor(x,t,u)
X_test,T_test,U_test = dataprocessing.reshape_data(X,T,U)
total_points=len(x[0])*len(t[0])
print('The dataset has',total_points,'points')
 

#Selection of dataset
Nf = 8100
X_data_tensor,T_data_tensor,U_data_tensor = dataprocessing.select_data(total_points,Nf,X_test,T_test,U_test)



#Add Latin hyper cube sampling
num_samples = 90
parameter_ranges = np.array([[-5, 5], [0, 100]])
samples = lhs(2, samples=num_samples, criterion='maximin', iterations=1000)
for i in range(2):
    samples[:, i] = samples[:, i] * (parameter_ranges[i, 1] - parameter_ranges[i, 0]) + parameter_ranges[i, 0]
x_samples = samples[:, 0]
t_samples = samples[:, 1]
X_physics,T_physics = np.meshgrid(x_samples,t_samples)
X_physics_tensor = torch.tensor(X_physics, dtype=torch.float32).view(-1,1)
T_physics_tensor = torch.tensor(T_physics, dtype=torch.float32).view(-1,1)


X_data_tensor.requires_grad = True
T_data_tensor.requires_grad = True
U_data_tensor.requires_grad = True
X_physics_tensor.requires_grad = True
T_physics_tensor.requires_grad = True

#Make Validation Set
# Nf_val = 500      #Num of validation set
# X_data_tensor_train,X_data_tensor_val = dataprocessing.split_data(X_data_tensor,Nf_val)
# T_data_tensor_train,T_data_tensor_val = dataprocessing.split_data(T_data_tensor,Nf_val)
# U_data_tensor_train,U_data_tensor_val = dataprocessing.split_data(U_data_tensor,Nf_val)
# X_physics_tensor_train,X_physics_tensor_val = dataprocessing.split_data(X_physics_tensor,Nf_val)
# T_physics_tensor_train,T_physics_tensor_val = dataprocessing.split_data(T_physics_tensor,Nf_val)
# U_physics_tensor_train,U_physics_tensor_val = dataprocessing.split_data(U_physics_tensor,Nf_val)
#plotdataphysics(X_data_tensor,T_data_tensor,X_physics_tensor,T_physics_tensor)


# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# surf = ax.plot_surface(X_data, T_data, U_data, cmap='viridis', edgecolor='none')
# ax.scatter(X_data,T_data,U_data,color='red', label='Data Points')
# fig.colorbar(surf, ax=ax, label='u(x, t)')
# ax.set_title('Surface Plot of u(x, t)')
# ax.set_xlabel('x')
# ax.set_ylabel('t')
# ax.set_zlabel('u(x, t)')
# plt.show()


# define a neural network to train
pinn = FCN(2,1,32,2)
fx = FNet(1,1,32,3)
gx = GNet(1,1,32,3)


# add mu to the optimiser
# TODO: write code here
# optimiser = torch.optim.Adam(list(pinn.parameters())+[alpha]+[beta]+[gamma],lr=1e-3)
# optimiser = torch.optim.Adam(list(pinn.parameters())+list(fx.parameters())+list(gx.parameters()),lr=1e-3)
optimiser1 = torch.optim.Adam(list(pinn.parameters()),lr=1e-3)
optimiser2 = torch.optim.Adam(list(fx.parameters())+list(gx.parameters()),lr=0.001)
writer = SummaryWriter()

theta = 0.01
loss = 1
i = 0
# for i in range(30001):
time_start = time.time()

try:
    while loss > theta:
        
        optimiser1.zero_grad()
        optimiser2.zero_grad()
        
        # compute each term of the PINN loss function above
        # using the following hyperparameters:
        lambda1 = 1e4
        
        # compute physics loss
        physics_input = [X_physics_tensor,T_physics_tensor]
        physic_output = pinn(physics_input)
        dudt = torch.autograd.grad(physic_output, T_physics_tensor, torch.ones_like(physic_output), create_graph=True)[0]
        dudx = torch.autograd.grad(physic_output, X_physics_tensor, torch.ones_like(physic_output), create_graph=True)[0]
        d2udx2 = torch.autograd.grad(dudx, X_physics_tensor, torch.ones_like(dudx), create_graph=True)[0]
        # loss1 = torch.mean((alpha*d2udx2+beta*dudx+gamma*physic_output-dudt)**2)
        co_f = fx(X_physics_tensor)
        co_g = gx(X_physics_tensor)
        loss1 = torch.mean((co_f*d2udx2+co_g*dudx-dudt)**2)
        # compute data loss
        # TODO: write code here
        data_input = [X_data_tensor,T_data_tensor]
        data_output = pinn(data_input)
        loss2 = torch.mean((U_data_tensor - data_output)**2)
        
        # backpropagate joint loss, take optimiser step
        loss = loss1 + lambda1*loss2
        loss.backward(retain_graph=True)
        optimiser1.step()
        optimiser2.step()
        
        # record mu value
        # TODO: write code here
        writer.add_scalar('train_loss',loss,i)

        # plot the result as training progresses

        # #The validation
        # physics_input_val = [X_physics_tensor_val,T_physics_tensor_val]
        # physic_output_val = pinn(physics_input_val)
        # dudt_v = torch.autograd.grad(physic_output_val, T_physics_tensor_val, torch.ones_like(physic_output_val), create_graph=True)[0]
        # dudx_v = torch.autograd.grad(physic_output_val, X_physics_tensor_val, torch.ones_like(physic_output_val), create_graph=True)[0]
        # d2udx2_v= torch.autograd.grad(dudx_v, X_physics_tensor_val, torch.ones_like(dudx_v), create_graph=True)[0]
        # # loss1 = torch.mean((alpha*d2udx2+beta*dudx+gamma*physic_output-dudt)**2)
        # loss1_v = torch.mean((alpha*d2udx2_v+beta*dudx_v+gamma*physic_output_val-dudt_v)**2)
        # # compute data loss
        # # TODO: write code here
        # data_input_val = [X_data_tensor_val,T_data_tensor_val]
        # data_output_val = pinn(data_input_val)
        # loss2_v = torch.mean((U_data_tensor_val - data_output_val)**2)
        # # backpropagate joint loss, take optimiser step
        # loss_v = loss1_v + lambda1*loss2_v
        # writer.add_scalar('val_loss',loss_v,i)



        if i % 500 == 0: 
            # u = pinn(t_test).detach()
            # plt.figure(figsize=(6,2.5))
            # plt.scatter(t_obs[:,0], u_obs[:,0], label="Noisy observations", alpha=0.6)
            # plt.plot(t_test[:,0], u[:,0], label="PINN solution", color="tab:green")
            # plt.title(f"Training step {i}")
            # plt.legend()
            # plt.show()
            # print(f'epoch: {i}  train loss :{loss} and validation loss:{loss_v}' )
            print(f'epoch: {i}  train loss :{loss}')
        i = i+1
except KeyboardInterrupt:
    print("Interrupted training loop.")

torch.save(pinn,"./sd_model/PINN_13100530.pkl.")
torch.save(fx,"./sd_model/FX_13100530.pkl.")
torch.save(gx,"./sd_model/GX_13100530.pkl.")
time_end = time.time()
time_sum = time_end - time_start
print('训练时间 {:.0f}分 {:.0f}秒'.format(time_sum // 60, time_sum % 60))


