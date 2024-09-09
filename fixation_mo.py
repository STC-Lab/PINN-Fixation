
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
import torch.optim as optim
import json
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris


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
    


torch.manual_seed(4321)
# first, create some noisy observational data
file = './dataset/3matrix_x20_t20.mat'
x,t,u1,u2 = dataprocessing.load_matrix(file)
X,T,U1,U2 = dataprocessing.matrix_totensor(x,t,u1,u2)
X_test,T_test,U1_test,U2_test = dataprocessing.reshape_matrix(X,T,U1,U2)
total_points=len(x[0])*len(t[0])
print('The dataset has',total_points,'points')
Nf =  400 # Nf: Number of collocation points 
X_data_tensor,T_data_tensor,U1_data_tensor,U2_data_tensor= dataprocessing.full_data_matrix(total_points,Nf,X_test,T_test,U1_test,U2_test)
# Nf =  int(total_points/2)

Nf_val =int(Nf*0.1)     #Num of validation set
X_data_tensor_train,X_data_tensor_val,T_data_tensor_train,T_data_tensor_val,U1_data_tensor_train,U1_data_tensor_val,U2_data_tensor_train,U2_data_tensor_val= dataprocessing.split_matrix(X_data_tensor,T_data_tensor,U1_data_tensor,U2_data_tensor,Nf_val)


#Add Latin hyper cube sampling
num_samples = 73  # 5000 physics points
parameter_ranges = np.array([[0, 1], [0,1]])
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
# X_bc_left_tensor.requires_grad = True
# X_bc_right_tensor.requires_grad = True

# plotdataphysics(X_data_tensor,T_data_tensor,X_physics_tensor,T_physics_tensor)




# define a neural network to train
pinn = FCN(2,2,32,3)
# pinn = FCN_2output(2,1,32,2)

alpha = torch.nn.Parameter(torch.randn(2, 2), requires_grad=True)
mu = torch.nn.Parameter(torch.randn(2,2), requires_grad=True)


alps1 = []
alps2 = []
alps3 = []
alps4 = []
mus1 = []
mus2 = []
mus3 = []
mus4 = []
physicsloss = []
dataloss = []
totalloss = []
valloss = []
error = []

# add mu to the optimiser
# TODO: write code here
# optimiser = torch.optim.Adam(list(pinn.parameters())+[alpha1]+[alpha2]+[alpha3]+[alpha4]+[alpha5],lr=0.0001)
optimiser = torch.optim.Adam(list(pinn.parameters())+[alpha]+[mu],lr=0.0001)
#writer = SummaryWriter()

# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', patience=2000, factor=0.1)
early_stop_epochs = 6000
theta = 0.0001
loss = 1
i = 0
epochs_no_improve = 0
best_loss = float('inf')
time_start = time.time()


try:
    # while loss > theta:
    for i in range(30001):
        
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


        #Physics loss
        loss1 = torch.mean((alpha[0][0]*d2u1dx2+alpha[0][1]*d2u2dx2+mu[0][0]*physic_output1+mu[0][1]*physic_output2-du1dt)**2+(alpha[1][0]*d2u1dx2+alpha[1][1]*d2u2dx2+mu[1][0]*physic_output1+mu[1][1]*physic_output2-du2dt)**2)
        physicsloss.append(loss1.item())
        #writer.add_scalar('loss1',loss1,i)

        

        # compute data loss
        # TODO: write code here
        data_input = [X_data_tensor_train,T_data_tensor_train]
        data_output = pinn(data_input)
        data_output1 = data_output[:, 0].view(-1, 1)
        data_output2 = data_output[:, 1].view(-1, 1)
        loss2 = torch.mean((U1_data_tensor_train - data_output1)**2+(U2_data_tensor_train-data_output2)**2)
        dataloss.append(loss2.item())
        #writer.add_scalar('loss2',loss2,i)



        # backpropagate joint loss, take optimiser step
        loss = loss1+lambda1*loss2
        paraerror = 1/8*(abs(alpha[0][0].item()-3)+abs(alpha[0][1]-(-1.1))+abs(alpha[1][0]-2.7)+abs(alpha[1][1]-1.5)+abs(mu[0][0]-2)+abs(mu[0][1]-1.8)+abs(mu[1][0]-1.2)+abs(mu[1][1]-(-2.5)))
        totalloss.append(loss.item())
        loss.backward()
        optimiser.step()
        # scheduler.step(loss)
        
        # record mu value
        # TODO: write code here
        alps1.append(alpha[0][0].item())
        alps2.append(alpha[0][1].item())
        alps3.append(alpha[1][0].item())
        alps4.append(alpha[1][1].item())
        mus1.append(mu[0][0].item())
        mus2.append(mu[0][1].item())
        mus3.append(mu[1][0].item())
        mus4.append(mu[1][1].item())
        error.append(paraerror.item())
        # alps5.append(alpha5.item())

        #validation
        physics_input_v = [X_data_tensor_val,T_data_tensor_val]
        physic_output_v = pinn(physics_input_v)
        physic_output1_v = physic_output_v[:, 0].view(-1, 1)
        physic_output2_v = physic_output_v[:, 1].view(-1, 1)
        du1dt_v = torch.autograd.grad(physic_output1_v, T_data_tensor_val, torch.ones_like(physic_output1_v), create_graph=True)[0]
        du2dt_v = torch.autograd.grad(physic_output2_v, T_data_tensor_val, torch.ones_like(physic_output2_v), create_graph=True)[0]
        du1dx_v = torch.autograd.grad(physic_output1_v, X_data_tensor_val, torch.ones_like(physic_output1_v), create_graph=True)[0]
        du2dx_v = torch.autograd.grad(physic_output2_v, X_data_tensor_val, torch.ones_like(physic_output2_v), create_graph=True)[0]
        d2u1dx2_v = torch.autograd.grad(du1dx_v, X_data_tensor_val, torch.ones_like(du1dx_v), create_graph=True)[0]
        d2u2dx2_v = torch.autograd.grad(du2dx_v, X_data_tensor_val, torch.ones_like(du2dx_v), create_graph=True)[0]

        #Physics loss
        loss1_v = torch.mean((alpha[0][0]*d2u1dx2_v+alpha[0][1]*d2u2dx2_v+mu[0][0]*physic_output1_v+mu[0][1]*physic_output2_v-du1dt_v)**2+(alpha[1][0]*d2u1dx2_v+alpha[1][1]*d2u2dx2_v+mu[1][0]*physic_output1_v+mu[1][1]*physic_output2_v-du2dt_v)**2)

        data_input_v = [X_data_tensor_val,T_data_tensor_val]
        data_output_v = pinn(data_input_v)
        data_output1_v = data_output_v[:, 0].view(-1, 1)
        data_output2_v = data_output_v[:, 1].view(-1, 1)
        loss2_v = torch.mean((U1_data_tensor_val - data_output1_v)**2+(U2_data_tensor_val-data_output2_v)**2)


        loss_v = loss1_v+lambda1*loss2_v
        valloss.append(loss_v.item())

            # 检查是否有改进
        if loss_v.item() < best_loss:
            best_loss = loss_v.item()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
    
        # 早停判断
        if epochs_no_improve == early_stop_epochs:
            print(f'Early stopping at epoch {i}')
            break


        if i % 500 == 0: 

             print(f'epoch: {i}  train loss :{loss}, valloss:{loss_v} ,alpha: {alpha},mu:{mu}' )
        # i = i+1
except KeyboardInterrupt:
    print("Interrupted training loop.")


minloss = min(valloss)
min_index = valloss.index(minloss)
minerror = error[min_index]

data = {'alpha1':alps1,
         'alpha2':alps2,
         'alpha3':alps3,
         'alpha4':alps4,
        'mu1':mus1,
        'mu2':mus2,
        'mu3':mus3,
        'mu4':mus4,
         'physicsloss':physicsloss,
         'dataloss':dataloss,
         'totalloss':totalloss,
         'valloss': valloss,
         'finalloss':minloss,
         'paraerror': error,
         'finalerror': minerror,
         'index': min_index,
}

with open('3matrix_lr0.0001.json',"w") as f:
    json.dump(data,f,indent=4)

torch.save(pinn,"./model/3matrix_lr0.0001.pkl")
time_end = time.time()
time_sum = time_end - time_start
print('训练时间 {:.0f}分 {:.0f}秒'.format(time_sum // 60, time_sum % 60))
        

