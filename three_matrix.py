
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from scipy.io import loadmat
import dataprocessing
import time
import numpy as np
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
    

torch.manual_seed(1234)
# first, create some noisy observational data
file = './dataset/3.5-5.5_8.5-12.5.mat'
x,t,u1,u2 = dataprocessing.load_matrix(file)
X,T,U1,U2 = dataprocessing.matrix_totensor(x,t,u1,u2)
X_test,T_test,U1_test,U2_test = dataprocessing.reshape_matrix(X,T,U1,U2)
total_points=len(x[0])*len(t[0])
print('The dataset has',total_points,'points')
Nf =  8100 # Nf: Number of collocation points 
X_data_tensor,T_data_tensor,U1_data_tensor,U2_data_tensor = dataprocessing.full_data_matrix(total_points,Nf,X_test,T_test,U1_test,U2_test)
# Nf =  int(total_points/2)
# X_data_tensor,T_data_tensor,U1_data_tensor,U2_data_tensor,X_physics_tensor,T_physics_tensor,U1_physics_tensor,U2_physics_tensor = dataprocessing.full_data_matrix(total_points,Nf,X_test,T_test,U1_test,U2_test)


#Linspace sampling
# Nf = 2500
# X_data_tensor,T_data_tensor,U1_data_tensor,U2_data_tensor = dataprocessing.select_random_matrix_data(total_points,Nf,X_test,T_test,U1_test,U2_test)
# t_physics = torch.linspace(0,1,50).view(-1,1)
# x_physics = torch.linspace(0,1,50).view(-1,1)
# X_physics,T_physics = np.meshgrid(x_physics,t_physics)
# X_physics_tensor = torch.tensor(X_physics, dtype=torch.float32).view(-1,1)
# T_physics_tensor = torch.tensor(T_physics, dtype=torch.float32).view(-1,1)

#Add Latin hyper cube sampling
num_samples = 90
parameter_ranges = np.array([[0, 1], [0, 1]])
samples = lhs(2, samples=num_samples, criterion='maximin', iterations=1000)
for i in range(2):
    samples[:, i] = samples[:, i] * (parameter_ranges[i, 1] - parameter_ranges[i, 0]) + parameter_ranges[i, 0]
x_samples = samples[:, 0]
t_samples = samples[:, 1]
X_physics,T_physics = np.meshgrid(x_samples,t_samples)
X_physics_tensor = torch.tensor(X_physics, dtype=torch.float32).view(-1,1)
T_physics_tensor = torch.tensor(T_physics, dtype=torch.float32).view(-1,1)


#IC
T_zero = np.zeros(90)
X_ic,T_ic = np.meshgrid(x_samples,T_zero)
X_ic_tensor = torch.tensor(X_ic, dtype=torch.float32).view(-1,1)
T_ic_tensor = torch.tensor(T_ic, dtype=torch.float32).view(-1,1)
#BC
X_zero = np.zeros(90)
X_one = np.ones(90)
X_bc_left,T_bc = np.meshgrid(X_zero,t_samples)
X_bc_left_tensor = torch.tensor(X_bc_left, dtype=torch.float32).view(-1,1)
T_bc_tensor = torch.tensor(T_bc, dtype=torch.float32).view(-1,1)
X_bc_right,T_bc = np.meshgrid(X_one,t_samples)
X_bc_right_tensor = torch.tensor(X_bc_right, dtype=torch.float32).view(-1,1)



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

# alpha1 = torch.nn.Parameter(torch.ones(1, requires_grad=True))
# beta1 = torch.nn.Parameter(torch.ones(1, requires_grad=True))
# gamma1 = torch.nn.Parameter(torch.ones(1, requires_grad=True))
# alpha2 = torch.nn.Parameter(torch.ones(1, requires_grad=True))
# beta2 = torch.nn.Parameter(torch.ones(1, requires_grad=True))
# gamma2 = torch.nn.Parameter(torch.ones(1, requires_grad=True))

alpha = torch.nn.Parameter(torch.randn(2), requires_grad=True)
# beta = torch.nn.Parameter(torch.randn(2), requires_grad=True)
gamma = torch.nn.Parameter(torch.randn(2), requires_grad=True)

# alpha1 = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
# beta1 = torch.nn.Parameter(torch.tensor(1.0),requires_grad=True)
# gamma1 = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
# alpha2 = torch.nn.Parameter(torch.tensor(1.0),requires_grad=True)
# beta2 = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
# gamma2 = torch.nn.Parameter(torch.tensor(1.0),requires_grad=True)

alps1 = []
alps2 = []
# bets1 = []
# bets2 = []
gams1 = []
gams2 = []


# add mu to the optimiser
# TODO: write code here
optimiser = torch.optim.Adam(list(pinn.parameters())+[alpha]+[gamma],lr=0.001)
writer = SummaryWriter()

theta = 0.01
loss = 1
i = 0

time_start = time.time()

try:
    while loss > theta:
    # for i in range(40001):
        
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
        loss1 = torch.mean((alpha[0]*d2u1dx2+gamma[0]*physic_output1-du1dt)**2+(alpha[1]*d2u2dx2+gamma[1]*physic_output1-du2dt)**2)
        # loss11 = torch.mean((alpha1*d2u1dx2+beta1*du1dx+gamma1*physic_output1-du1dt)**2)
        # loss12 = torch.mean((alpha2*d2u2dx2+beta2*du2dx+gamma2*physic_output2-du2dt)**2)
        writer.add_scalar('loss1',loss1,i)
        # writer.add_scalar('loss12',loss12,i)
        

        # compute data loss
        # TODO: write code here
        data_input = [X_data_tensor,T_data_tensor]
        data_output = pinn(data_input)
        data_output1 = data_output[:, 0].view(-1, 1)
        data_output2 = data_output[:, 1].view(-1, 1)
        loss2 = torch.mean((U1_data_tensor - data_output1)**2+(U2_data_tensor-data_output2)**2)
        # loss21 = torch.mean((U1_data_tensor - data_output1)**2)
        # loss22 = torch.mean((U2_data_tensor-data_output2)**2)
        writer.add_scalar('loss2',loss2,i)
        # writer.add_scalar('loss22',loss22,i)


        #Initial Condition
        physics_input = [X_ic_tensor,T_ic_tensor]
        physic_output = pinn(physics_input)
        physic_output1 = physic_output[:, 0].view(-1, 1)
        physic_output2 = physic_output[:, 1].view(-1, 1)
        loss3 = torch.mean((physic_output1-np.sin(0.5*np.pi*X_ic_tensor))**2+(physic_output2-np.sin(0.5*np.pi*X_ic_tensor))**2)
        writer.add_scalar('loss3',loss3,i)


        # backpropagate joint loss, take optimiser step
        # loss1 = loss11 + lambda1*loss21
        # loss2 = loss12 + lambda1*loss22
        loss = loss1 + lambda1*(loss2+loss3)
        loss.backward()
        optimiser.step()
        
        # record mu value
        # TODO: write code here
        alps1.append(alpha[0].item())
        alps2.append(alpha[1].item())
        # bets1.append(beta[0].item())
        # bets2.append(beta[1].item())
        gams1.append(gamma[0].item())
        gams2.append(gamma[1].item())
        writer.add_scalar('train_loss',loss,i)
        writer.add_scalar('alpha1',alpha[0],i)
        writer.add_scalar('alpha2',alpha[1],i)
        # writer.add_scalar('beta1',beta[0],i)
        # writer.add_scalar('beta2',beta[1],i)
        writer.add_scalar('gamma1',gamma[0],i)
        writer.add_scalar('gamma2',gamma[1],i)
        # plot the result as training progresses
        if i % 500 == 0: 
            # u = pinn(t_test).detach()
            # plt.figure(figsize=(6,2.5))
            # plt.scatter(t_obs[:,0], u_obs[:,0], label="Noisy observations", alpha=0.6)
            # plt.plot(t_test[:,0], u[:,0], label="PINN solution", color="tab:green")
            # plt.title(f"Training step {i}")
            # plt.legend()
            # plt.show()
            # print(f'epoch: {i}  train loss :{loss}, alpha: {alpha[0].item(),alpha[1].item()},beta:{beta[0].item(),beta[1].item()},gamma:{gamma[0].item(),gamma[1].item()}' )
             print(f'epoch: {i}  train loss :{loss}, alpha: {alpha[0].item(),alpha[1].item()},gamma:{gamma[0].item(),gamma[1].item()}' )
        i = i+1
except KeyboardInterrupt:
    print("Interrupted training loop.")


        
torch.save(pinn,"./model/18010502.pkl.")
time_end = time.time()
time_sum = time_end - time_start
print('训练时间 {:.0f}分 {:.0f}秒'.format(time_sum // 60, time_sum % 60))
        
plt.figure()
plt.title("alpha_1")
plt.plot(alps1, label="PINN estimate")
plt.hlines(3.5, 0, len(alps1), label="True value", color="tab:green")
plt.legend()
plt.xlabel("Training step")
plt.show()

plt.figure()
plt.title("alpha_2")
plt.plot(alps2, label="PINN estimate")
plt.hlines(5.5, 0, len(alps2), label="True value", color="tab:green")
plt.legend()
plt.xlabel("Training step")
plt.show()

# plt.figure()
# plt.title("beta_1")
# plt.plot(bets1, label="PINN estimate")
# plt.hlines(4.7, 0, len(bets1), label="True value", color="tab:green")
# plt.legend()
# plt.xlabel("Training step")
# plt.show()

# plt.figure()
# plt.title("beta_2")
# plt.plot(bets2, label="PINN estimate")
# plt.hlines(4.7, 0, len(bets2), label="True value", color="tab:green")
# plt.legend()
# plt.xlabel("Training step")
# plt.show()

plt.figure()
plt.title("gamma_1")
plt.plot(gams1, label="PINN estimate")
plt.hlines(8.5, 0, len(gams1), label="True value", color="tab:green")
plt.legend()
plt.xlabel("Training step")
plt.show()

plt.figure()
plt.title("gamma_2")
plt.plot(gams2, label="PINN estimate")
plt.hlines(0, 0, len(gams2), label="True value", color="tab:green")
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