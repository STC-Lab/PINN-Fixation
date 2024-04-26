
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from scipy.io import loadmat
import dataprocessing
import time


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
    

class FCN_2output(nn.Module):
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
        self.fce1 = nn.Linear(N_HIDDEN, N_OUTPUT)
        self.fce2 = nn.Linear(N_HIDDEN, N_OUTPUT)

    def forward(self, input):
        x = torch.cat(input, dim=-1)
        x = self.fcs(x)
        x = self.fch(x)
        u1 = self.fce1(x)
        u2 = self.fce2(x)
        return u1,u2
    

# torch.manual_seed(1234)
torch.manual_seed(123)
# first, create some noisy observational data
file = './dataset/3matrix_2-5_3-6_4-7.mat'
x,t,u1,u2 = dataprocessing.load_matrix(file)
X,T,U1,U2 = dataprocessing.matrix_totensor(x,t,u1,u2)
X_test,T_test,U1_test,U2_test = dataprocessing.reshape_matrix(X,T,U1,U2)
total_points=len(x[0])*len(t[0])
print('The dataset has',total_points,'points')
# Nf =  10 # Nf: Number of collocation points 
Nf =  int(total_points/2)
X_data_tensor,T_data_tensor,U1_data_tensor,U2_data_tensor,X_physics_tensor,T_physics_tensor,U1_physics_tensor,U2_physics_tensor = dataprocessing.full_data_matrix(total_points,Nf,X_test,T_test,U1_test,U2_test)


# Nf = 2500
# X_data_tensor,T_data_tensor,U1_data_tensor,U2_data_tensor = dataprocessing.select_random_matrix_data(total_points,Nf,X_test,T_test,U1_test,U2_test)
# t_physics = torch.linspace(0,1,50).view(-1,1)
# x_physics = torch.linspace(0,1,50).view(-1,1)
# X_physics,T_physics = np.meshgrid(x_physics,t_physics)
# X_physics_tensor = torch.tensor(X_physics, dtype=torch.float32).view(-1,1)
# T_physics_tensor = torch.tensor(T_physics, dtype=torch.float32).view(-1,1)


X_data_tensor.requires_grad = True
T_data_tensor.requires_grad = True
U1_data_tensor.requires_grad = True
U2_data_tensor.requires_grad = True
X_physics_tensor.requires_grad = True
T_physics_tensor.requires_grad = True


# plotdataphysics(X_data_tensor,T_data_tensor,X_physics_tensor,T_physics_tensor)




# define a neural network to train
pinn = FCN(2,2,32,2)
# pinn = FCN_2output(2,1,32,2)

alpha1 = torch.nn.Parameter(torch.ones(1, requires_grad=True))
beta1 = torch.nn.Parameter(torch.ones(1, requires_grad=True))
gamma1 = torch.nn.Parameter(torch.ones(1, requires_grad=True))
alpha2 = torch.nn.Parameter(torch.ones(1, requires_grad=True))
beta2 = torch.nn.Parameter(torch.ones(1, requires_grad=True))
gamma2 = torch.nn.Parameter(torch.ones(1, requires_grad=True))

alps1 = []
alps2 = []
bets1 = []
bets2 = []
gams1 = []
gams2 = []


# add mu to the optimiser
# TODO: write code here
optimiser = torch.optim.Adam(list(pinn.parameters())+[alpha1]+[beta1]+[gamma1]+[alpha2]+[beta2]+[gamma2],lr=0.001)
writer = SummaryWriter()

# theta = 0.1
# loss = 1
# i = 0

time_start = time.time()

# while loss > theta:
for i in range(20001):
    
    optimiser.zero_grad()
    
    # compute each term of the PINN loss function above
    # using the following hyperparameters:
    lambda1 = 1e4
    
    # compute physics loss
    physics_input = [X_physics_tensor,T_physics_tensor]
    physic_output = pinn(physics_input)
    physic_output1 = physic_output[:, 0].view(-1, 1)
    physic_output2 = physic_output[:, 1].view(-1, 1)

    du1dt = torch.autograd.grad(physic_output1, T_physics_tensor, torch.ones_like(physic_output1), create_graph=True)[0]
    du2dt = torch.autograd.grad(physic_output2, T_physics_tensor, torch.ones_like(physic_output2), create_graph=True)[0]
    du1dx = torch.autograd.grad(physic_output1, X_physics_tensor, torch.ones_like(physic_output1), create_graph=True)[0]
    du2dx = torch.autograd.grad(physic_output2, X_physics_tensor, torch.ones_like(physic_output2), create_graph=True)[0]
    d2u1dx2 = torch.autograd.grad(du1dx, X_physics_tensor, torch.ones_like(du1dx), create_graph=True)[0]
    d2u2dx2 = torch.autograd.grad(du2dx, X_physics_tensor, torch.ones_like(du2dx), create_graph=True)[0]
    # loss1 = torch.mean((alpha1*d2u1dx2+beta1*du1dx+gamma1*physic_output1-du1dt)**2+(alpha2*d2u2dx2+beta2*du2dx+gamma2*physic_output2-du2dt)**2)
    # loss1 = torch.mean((alpha[0]*d2u1dx2+beta[0]*du1dx-du1dt)**2+(alpha[1]*d2u2dx2+beta[1]*du2dx-du2dt)**2)
    loss1 = torch.mean((alpha1*d2u1dx2+beta1*du1dx+gamma1*physic_output1-du1dt)**2+(alpha2*d2u2dx2+beta2*du2dx+gamma2*physic_output2-du2dt)**2)
    writer.add_scalar('loss1',loss1,i)
    
    # compute data loss
    # TODO: write code here
    data_input = [X_data_tensor,T_data_tensor]
    data_output = pinn(data_input)
    data_output1 = data_output[:, 0].view(-1, 1)
    data_output2 = data_output[:, 1].view(-1, 1)
    loss2 = torch.mean((U1_data_tensor - data_output1)**2+(U2_data_tensor-data_output2)**2)
    writer.add_scalar('loss2',loss2,i)

    # backpropagate joint loss, take optimiser step
    loss = loss1 + lambda1*loss2
    loss.backward()
    optimiser.step()
    
    # record mu value
    # TODO: write code here
    alps1.append(alpha1.item())
    alps2.append(alpha2.item())
    bets1.append(beta1.item())
    bets2.append(beta2.item())
    gams1.append(gamma1.item())
    gams2.append(gamma2.item())
    writer.add_scalar('train_loss',loss,i)
    writer.add_scalar('alpha1',alpha1,i)
    writer.add_scalar('alpha2',alpha2,i)
    writer.add_scalar('beta1',beta1,i)
    writer.add_scalar('beta2',beta2,i)
    writer.add_scalar('gamma1',gamma1,i)
    writer.add_scalar('gamma2',gamma2,i)
    # plot the result as training progresses
    if i % 500 == 0: 
        # u = pinn(t_test).detach()
        # plt.figure(figsize=(6,2.5))
        # plt.scatter(t_obs[:,0], u_obs[:,0], label="Noisy observations", alpha=0.6)
        # plt.plot(t_test[:,0], u[:,0], label="PINN solution", color="tab:green")
        # plt.title(f"Training step {i}")
        # plt.legend()
        # plt.show()
        print(f'epoch: {i}  train loss :{loss}, alpha: {alpha1.item(),alpha2.item()},beta:{beta1.item(),beta2.item()},gamma:{gamma1.item(),gamma2.item()}' )
    # i = i+1

        
torch.save(pinn,"./model/18170426.pkl.")
time_end = time.time()
time_sum = time_end - time_start
print('训练时间 {:.0f}分 {:.0f}秒'.format(time_sum // 60, time_sum % 60))
        
plt.figure()
plt.title("alpha_1")
plt.plot(alps1, label="PINN estimate")
plt.hlines(2, 0, len(alps1), label="True value", color="tab:green")
plt.legend()
plt.xlabel("Training step")
plt.show()

plt.figure()
plt.title("alpha_2")
plt.plot(alps2, label="PINN estimate")
plt.hlines(5, 0, len(alps2), label="True value", color="tab:green")
plt.legend()
plt.xlabel("Training step")
plt.show()

plt.figure()
plt.title("beta_1")
plt.plot(bets1, label="PINN estimate")
plt.hlines(3, 0, len(bets1), label="True value", color="tab:green")
plt.legend()
plt.xlabel("Training step")
plt.show()

plt.figure()
plt.title("beta_2")
plt.plot(bets2, label="PINN estimate")
plt.hlines(6, 0, len(bets2), label="True value", color="tab:green")
plt.legend()
plt.xlabel("Training step")
plt.show()

plt.figure()
plt.title("gamma_1")
plt.plot(gams1, label="PINN estimate")
plt.hlines(4, 0, len(gams1), label="True value", color="tab:green")
plt.legend()
plt.xlabel("Training step")
plt.show()

plt.figure()
plt.title("gamma_2")
plt.plot(gams2, label="PINN estimate")
plt.hlines(7, 0, len(gams2), label="True value", color="tab:green")
plt.legend()
plt.xlabel("Training step")
plt.show()

# plt.figure()
# plt.title("gamma")
# plt.plot(gams, label="PINN estimate")
# plt.hlines(8.5, 0, len(gams), label="True value", color="tab:green")
# plt.legend()
# plt.xlabel("Training step")
# plt.show()