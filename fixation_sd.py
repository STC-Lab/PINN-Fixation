
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from scipy.io import loadmat
import dataprocessing
import time
from pyDOE import lhs
import torch.optim as optim
import json



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
file = './dataset/1to1_solver.mat'
x,t,u = dataprocessing.load_data(file)
X,T,U = dataprocessing.totensor(x,t,u)
X_test,T_test,U_test = dataprocessing.reshape_data(X,T,U)
total_points=len(x[0])*len(t[0])
print('The dataset has',total_points,'points')
 

#Selection of dataset
Nf = 5000
X_data_tensor,T_data_tensor,U_data_tensor = dataprocessing.select_data(total_points,Nf,X_test,T_test,U_test)

Nf_val =int(Nf*0.1)     #Num of validation set
X_data_tensor_train,X_data_tensor_val,T_data_tensor_train,T_data_tensor_val,U_data_tensor_train,U_data_tensor_val= dataprocessing.split_data(X_data_tensor,T_data_tensor,U_data_tensor,Nf_val)

#Add Latin hyper cube sampling
num_samples = 71
parameter_ranges = np.array([[0, 1], [0, 1]])
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
X_data_tensor_val.requires_grad = True
T_data_tensor_val.requires_grad = True

# #IC
# T_zero = np.zeros(90)
# X_ic,T_ic = np.meshgrid(x_samples,T_zero)
# X_ic_tensor = torch.tensor(X_ic, dtype=torch.float32).view(-1,1)
# T_ic_tensor = torch.tensor(T_ic, dtype=torch.float32).view(-1,1)
# #BC
# X_zero = np.zeros(90)
# X_one = np.ones(90)
# X_bc_left,T_bc = np.meshgrid(X_zero,t_samples)
# X_bc_left_tensor = torch.tensor(X_bc_left, dtype=torch.float32).view(-1,1)
# T_bc_tensor = torch.tensor(T_bc, dtype=torch.float32).view(-1,1)
# X_bc_right,T_bc = np.meshgrid(X_zero,t_samples)
# X_bc_right_tensor = torch.tensor(X_bc_right, dtype=torch.float32).view(-1,1)



# define a neural network to train
pinn = FCN(2,1,32,3)
fx = FNet(1,1,32,3)
gx = GNet(1,1,32,3)
physicsloss = []
dataloss = []
totalloss = []
valloss = []
#icloss = []

# add mu to the optimiser
# TODO: write code here
optimiser = torch.optim.Adam(list(pinn.parameters())+list(fx.parameters())+list(gx.parameters()),lr=1e-4)
writer = SummaryWriter()

theta = 0.0001
loss = 1
i = 0



# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10000, factor=0.1)
# for i in range(30001):
time_start = time.time()

try:
    # for i in range(80001):
    while loss > theta:
        
        optimiser.zero_grad()
        
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
        # loss1 = torch.mean((co_f*d2udx2+co_g*dudx-dudt)**2)
        loss1 = torch.mean((co_f*d2udx2+co_g*dudx+10*torch.sin(10*T_physics_tensor)-dudt)**2)
        physicsloss.append(loss1.item())



        # compute data loss
        # TODO: write code here
        data_input = [X_data_tensor,T_data_tensor]
        data_output = pinn(data_input)
        loss2 = torch.mean((U_data_tensor - data_output)**2)
        dataloss.append(loss2.item())


        # #Initial Condition
        # ic_input = [X_ic_tensor,T_ic_tensor]
        # ic_output = pinn(ic_input)
        # loss3 = torch.mean((ic_output-1)**2)
        # icloss.append(loss3.item())
 



        # backpropagate joint loss, take optimiser step
        loss = loss1 + lambda1*loss2
        totalloss.append(loss.item())
        loss.backward(retain_graph=True)
        optimiser.step()
        
        # record mu value
        writer.add_scalar('train_loss',loss,i)


        #The validation
        physics_input_val = [X_data_tensor_val,T_data_tensor_val]
        physic_output_val = pinn(physics_input_val)
        dudt_v = torch.autograd.grad(physic_output_val, T_data_tensor_val, torch.ones_like(physic_output_val), create_graph=True)[0]
        dudx_v = torch.autograd.grad(physic_output_val, X_data_tensor_val, torch.ones_like(physic_output_val), create_graph=True)[0]
        d2udx2_v= torch.autograd.grad(dudx_v, X_data_tensor_val, torch.ones_like(dudx_v), create_graph=True)[0]
        # loss1 = torch.mean((alpha*d2udx2+beta*dudx+gamma*physic_output-dudt)**2)
        co_f_v = fx(X_data_tensor_val)
        co_g_v = gx(X_data_tensor_val)
        # loss1_v = torch.mean((co_f_v*d2udx2_v+co_g_v*dudx_v-dudt_v)**2)
        loss1_v = torch.mean((co_f_v*d2udx2_v+co_g_v*dudx_v+10*torch.sin(10*T_data_tensor_val)-dudt_v)**2)
        # compute data loss
        # TODO: write code here
        data_input_val = [X_data_tensor_val,T_data_tensor_val]
        data_output_val = pinn(data_input_val)
        loss2_v = torch.mean((U_data_tensor_val - data_output_val)**2)
        # backpropagate joint loss, take optimiser step
        loss_v = loss1_v + lambda1*loss2_v
        writer.add_scalar('val_loss',loss_v,i)
        valloss.append(loss_v.item())



        if i % 500 == 0: 
            print(f'epoch: {i}  train loss :{loss} val loss:{loss_v}')
        i = i+1
except KeyboardInterrupt:
    print("Interrupted training loop.")
        

time_end = time.time()
time_sum = time_end - time_start
print('训练时间 {:.0f}分 {:.0f}秒'.format(time_sum // 60, time_sum % 60))

data = {
        #  'alpha':alps,
         'physicsloss':physicsloss,
         'dataloss':dataloss,
         'totalloss':totalloss,
         'valloss':valloss,
         'time':time_sum,
}

with open('fixation_1to1_tanh_pretrained.json',"w") as f:
    json.dump(data,f,indent=4)




torch.save(pinn,"./sd_model/PINN_fixation_1to1_tanh_pretrained.pkl.")
torch.save(fx,"./sd_model/FX_fixation_1to1_tanh_pretrained.pkl.")
torch.save(gx,"./sd_model/GX_fixation_1to1_tanh_pretrained.pkl.")
time_end = time.time()
time_sum = time_end - time_start
print('训练时间 {:.0f}分 {:.0f}秒'.format(time_sum // 60, time_sum % 60))


