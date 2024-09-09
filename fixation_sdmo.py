
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
        activation = nn.ReLU
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
file = './dataset/2to2_solver2.mat'
x,t,u1,u2 = dataprocessing.load_matrix(file)
X,T,U1,U2 = dataprocessing.matrix_totensor(x,t,u1,u2)
X_test,T_test,U1_test,U2_test = dataprocessing.reshape_matrix(X,T,U1,U2)
total_points=len(x[0])*len(t[0])
print('The dataset has',total_points,'points')
Nf =  5000 # Nf: Number of collocation points 
X_data_tensor,T_data_tensor,U1_data_tensor,U2_data_tensor= dataprocessing.full_data_matrix(total_points,Nf,X_test,T_test,U1_test,U2_test)
# Nf =  int(total_points/2)

Nf_val =int(Nf*0.1)     #Num of validation set
X_data_tensor_train,X_data_tensor_val,T_data_tensor_train,T_data_tensor_val,U1_data_tensor_train,U1_data_tensor_val,U2_data_tensor_train,U2_data_tensor_val= dataprocessing.split_matrix(X_data_tensor,T_data_tensor,U1_data_tensor,U2_data_tensor,Nf_val)

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
U1_data_tensor.requires_grad = True
U2_data_tensor.requires_grad = True
X_physics_tensor.requires_grad = True
T_physics_tensor.requires_grad = True
X_data_tensor_val.requires_grad = True
T_data_tensor_val.requires_grad = True


# define a neural network to train
pinn = FCN(2,2,32,3)
fx = FNet(1,4,64,3)
gx = GNet(1,4,64,3)
physicsloss = []
dataloss = []
totalloss = []
valloss = []
#icloss = []

# add mu to the optimiser
# TODO: write code here
optimiser = torch.optim.Adam(list(pinn.parameters())+list(fx.parameters())+list(gx.parameters()),lr=1e-3)
writer = SummaryWriter()

theta = 0.00001
loss = 1
i = 0



# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10000, factor=0.1)
# for i in range(30001):
time_start = time.time()

try:
    # for i in range(80001):
    while loss > theta:
        
        optimiser.zero_grad()
        
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

        co_f = fx(X_physics_tensor)
        co_g = gx(X_physics_tensor)
        #Physics loss
        loss1 = torch.mean(abs(co_f[:,0].view(-1, 1)*d2u1dx2+co_f[:,1].view(-1, 1)*d2u2dx2+co_g[:,0].view(-1, 1)*du1dx+co_g[:,1].view(-1, 1)*du2dx+10*torch.sin(10*T_physics_tensor)-du1dt)
                             +abs(co_f[:,2].view(-1, 1)*d2u1dx2+co_f[:,3].view(-1, 1)*d2u2dx2+co_g[:,2].view(-1, 1)*du1dx+co_g[:,3].view(-1, 1)*du2dx+10*torch.sin(10*T_physics_tensor)-du2dt))
        # physicsloss.append(loss1.item())
        physicsloss.append(loss1.item())
        #writer.add_scalar('loss1',loss1,i)



        data_input = [X_data_tensor_train,T_data_tensor_train]
        data_output = pinn(data_input)
        data_output1 = data_output[:, 0].view(-1, 1)
        data_output2 = data_output[:, 1].view(-1, 1)
        loss2 = torch.mean((U1_data_tensor_train - data_output1)**2+(U2_data_tensor_train-data_output2)**2)
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
        physic_output1_val = physic_output_val[:, 0].view(-1, 1)
        physic_output2_val = physic_output_val[:, 1].view(-1, 1)
        du1dt_v = torch.autograd.grad(physic_output1_val, T_data_tensor_val, torch.ones_like(physic_output1_val), create_graph=True)[0]
        du2dt_v = torch.autograd.grad(physic_output2_val, T_data_tensor_val, torch.ones_like(physic_output2_val), create_graph=True)[0]
        du1dx_v = torch.autograd.grad(physic_output1_val, X_data_tensor_val, torch.ones_like(physic_output1_val), create_graph=True)[0]
        du2dx_v = torch.autograd.grad(physic_output2_val, X_data_tensor_val, torch.ones_like(physic_output2_val), create_graph=True)[0]
        d2u1dx2_v = torch.autograd.grad(du1dx_v, X_data_tensor_val, torch.ones_like(du1dx_v), create_graph=True)[0]
        d2u2dx2_v = torch.autograd.grad(du2dx_v, X_data_tensor_val, torch.ones_like(du2dx_v), create_graph=True)[0]
        # loss1 = torch.mean((alpha*d2udx2+beta*dudx+gamma*physic_output-dudt)**2)
        co_f_v = fx(X_data_tensor_val)
        co_g_v = gx(X_data_tensor_val)
        loss1_v = torch.mean(abs(co_f_v[:,0].view(-1, 1)*d2u1dx2_v+co_f_v[:,1].view(-1, 1)*d2u2dx2_v+co_g_v[:,0].view(-1, 1)*du1dx_v+co_g_v[:,1].view(-1, 1)*du2dx_v+10*torch.sin(10*T_data_tensor_val)-du1dt_v)
                             +abs(co_f_v[:,2].view(-1, 1)*d2u1dx2_v+co_f_v[:,3].view(-1, 1)*d2u2dx2_v+co_g_v[:,2].view(-1, 1)*du1dx_v+co_g_v[:,3].view(-1, 1)*du2dx_v+10*torch.sin(10*T_data_tensor_val)-du2dt_v))
        # physicsloss.append(loss1.item())
        # compute data loss
        # TODO: write code here

        data_input_v = [X_data_tensor_val,T_data_tensor_val]
        data_output_v = pinn(data_input_v)
        data_output1_v = data_output_v[:, 0].view(-1, 1)
        data_output2_v = data_output_v[:, 1].view(-1, 1)
        loss2_v = torch.mean((U1_data_tensor_val - data_output1_v)**2+(U2_data_tensor_val-data_output2_v)**2)
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

with open('solver2_2to2_tanh_64_3.json',"w") as f:
    json.dump(data,f,indent=4)




torch.save(pinn,"./sdmo_model/PINN_solver2_2to2_tanh_64_3.pkl.")
torch.save(fx,"./sdmo_model/FX_solver2_2to2_tanh_64_3.pkl.")
torch.save(gx,"./sdmo_model/GX_solver2_2to2_tanh_64_3.pkl.")
time_end = time.time()
time_sum = time_end - time_start
print('训练时间 {:.0f}分 {:.0f}秒'.format(time_sum // 60, time_sum % 60))


