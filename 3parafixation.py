
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


torch.set_default_dtype(torch.float)
torch.manual_seed(1234)
np.random.seed(1231)
# first, create some noisy observational data

#Create data points
file = './dataset111/3para_x45_t100.mat'        #The dataset includes x,t and u
x,t,u = dataprocessing.load_data(file)
X,T,U = dataprocessing.totensor(x,t,u)
X_test,T_test,U_test = dataprocessing.reshape_data(X,T,U)
total_points=len(x[0])*len(t[0])
print('The dataset has',total_points,'points')
 

#Selection of dataset
#Nf = 2000
# X_data_tensor,T_data_tensor,U_data_tensor = dataprocessing.select_data(total_points,Nf,X_test,T_test,U_test)

#Entire dataset training
Nf = 4500 # Nf: Number of collocation points
X_data_tensor,T_data_tensor,U_data_tensor = dataprocessing.select_data(total_points,Nf,X_test,T_test,U_test)



Nf_val =int(Nf*0.1)     #Num of validation set
X_data_tensor_train,X_data_tensor_val,T_data_tensor_train,T_data_tensor_val,U_data_tensor_train,U_data_tensor_val= dataprocessing.split_data(X_data_tensor,T_data_tensor,U_data_tensor,Nf_val)



#Add Latin hyper cube sampling
#Create Physics points
num_samples = 63
parameter_ranges = np.array([[0, 2], [0,1]])
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




#IC
T_zero = np.zeros(100)
X_ic,T_ic = np.meshgrid(x,T_zero)
X_ic_tensor = torch.tensor(X_ic, dtype=torch.float32).view(-1,1)
T_ic_tensor = torch.tensor(T_ic, dtype=torch.float32).view(-1,1)


# define a neural network to train
pinn = FCN(2,1,32,3)
alpha = torch.nn.Parameter(torch.ones(1, requires_grad=True))
beta = torch.nn.Parameter(torch.ones(1, requires_grad=True))
gamma = torch.nn.Parameter(torch.ones(1, requires_grad=True))
lambda1 = torch.nn.Parameter(torch.tensor(1.0, requires_grad=True))
# lambda2 = torch.nn.Parameter(torch.tensor(10000.0, requires_grad=True))
alps = []
bets = []
gams = []
physicsloss = []
dataloss = []
totalloss = []
valloss = []
lamb=[]
lamb2=[]
error = []

# add mu to the optimiser
# TODO: write code here
optimiser1 = torch.optim.Adam(list(pinn.parameters())+[alpha]+[beta]+[gamma],lr=0.001)
optimiser2 = torch.optim.Adam([lambda1],lr=0.01)
# optimiser = torch.optim.Adam(list(pinn.parameters())+[alpha]+[gamma],lr=1e-3)
writer = SummaryWriter()


early_stop_epochs = 6000
theta = 0.0001           #The threshhold of loss, if loss less than theta,the training stop
loss = 1
i = 0
epochs_no_improve = 0
best_loss = float('inf')
time_start = time.time()


try:
    while loss > theta:
    #for i in range(50001):    
        optimiser1.zero_grad()
        # optimiser2.zero_grad()
        
        # compute each term of the PINN loss function above
        # using the following hyperparameters:

        lambda2 = 1e3
        
        # compute physics loss
        physics_input = [X_physics_tensor,T_physics_tensor]
        physic_output = pinn(physics_input)
        dudt = torch.autograd.grad(physic_output, T_physics_tensor, torch.ones_like(physic_output), create_graph=True)[0]
        dudx = torch.autograd.grad(physic_output, X_physics_tensor, torch.ones_like(physic_output), create_graph=True)[0]
        d2udx2 = torch.autograd.grad(dudx, X_physics_tensor, torch.ones_like(dudx), create_graph=True)[0]
        loss1 = torch.mean((alpha*d2udx2+beta*dudx+gamma*physic_output-dudt)**2)
        physicsloss.append(loss1.item())



        # compute data loss
        # TODO: write code here
        data_input = [X_data_tensor_train,T_data_tensor_train]
        data_output = pinn(data_input)
        loss2 = torch.mean((U_data_tensor_train - data_output)**2)
        dataloss.append(loss2.item())



        loss = lambda1*loss1 + lambda2*loss2
        paraerror = 1/3*(abs(alpha-3)+abs(beta-6.7)+abs(gamma-8.5))
        totalloss.append(loss.item())
        loss.backward(retain_graph=True)
        lambda1.grad.data = -lambda1.grad.data
        # lambda2.grad.data = -lambda2.grad.data
        optimiser1.step()
        optimiser2.step()
        

        # optimiser2.zero_grad()
        # loss1.backward()
        # optimiser2.step()
        # record mu value
        # TODO: write code here
        alps.append(alpha.item())
        bets.append(beta.item())
        gams.append(gamma.item())
        lamb.append(lambda1.item())
        # lamb2.append(lambda2.item())
        error.append(paraerror.item())
        # writer.add_scalar('train_loss',loss,i)

        # plot the result as training progresses

        #The validation
        physics_input_val = [X_data_tensor_val,T_data_tensor_val]
        physic_output_val = pinn(physics_input_val)
        dudt_v = torch.autograd.grad(physic_output_val, T_data_tensor_val, torch.ones_like(physic_output_val), create_graph=True)[0]
        dudx_v = torch.autograd.grad(physic_output_val, X_data_tensor_val, torch.ones_like(physic_output_val), create_graph=True)[0]
        d2udx2_v= torch.autograd.grad(dudx_v, X_data_tensor_val, torch.ones_like(dudx_v), create_graph=True)[0]
        # loss1 = torch.mean((alpha*d2udx2+beta*dudx+gamma*physic_output-dudt)**2)
        loss1_v = torch.mean((alpha*d2udx2_v+beta*dudx_v+gamma*physic_output_val-dudt_v)**2)
        # compute data loss
        # TODO: write code here
        data_input_val = [X_data_tensor_val,T_data_tensor_val]
        data_output_val = pinn(data_input_val)
        loss2_v = torch.mean((U_data_tensor_val - data_output_val)**2)
        # backpropagate joint loss, take optimiser step
        # loss_v = loss1_v + lambda1*loss2_v
        # loss_v = lambda1*loss1_v + loss2_v
        loss_v = lambda1*loss1_v + lambda2*loss2_v
        # writer.add_scalar('val_loss',loss_v,i)
        valloss.append(loss_v.item())


        if loss_v.item() < best_loss:
            best_loss = loss_v.item()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
    

        if epochs_no_improve == early_stop_epochs:
            print(f'Early stopping at epoch {i}')
            break



        if i % 500 == 0: 
            print(f'epoch: {i}  train loss :{loss} val loss:{loss_v} alpha: {alpha} beta:{beta} gamma:{gamma} lambda:{lambda1,lambda2}')
            # print(f'epoch: {i}  train loss :{loss} alpha: {alpha} gamma:{gamma}')
        i = i+1
except KeyboardInterrupt:
    print("Interrupted training loop.")



minloss = min(valloss)
min_index = valloss.index(minloss)
minerror = error[min_index]


time_end = time.time()
time_sum = time_end - time_start
print('训练时间 {:.0f}分 {:.0f}秒'.format(time_sum // 60, time_sum % 60))

data = {'alpha':alps,
        'beta':bets,
        'gamma':gams,
        'lambda':lamb,
        # 'lambda2':lamb2,
         'physicsloss':physicsloss,
         'dataloss':dataloss,
         'totalloss':totalloss,
         'valloss':valloss,
         'time':time_sum,
        'finalloss':minloss,
         'paraerror': error,
         'finalerror': minerror,
         'index': min_index,
}

with open('3para_phylamb0.01_datalamb1000_init1.json',"w") as f:
    json.dump(data,f,indent=4)

torch.save(pinn,"./model/3para_phylamb0.01_datalamb1000_init1.pkl.")


# plt.figure()
# plt.title("alpha")
# plt.plot(alps, label="PINN estimate")
# plt.hlines(3, 0, len(alps), label="True value", color="tab:green")
# plt.legend()
# plt.xlabel("Training step")
# plt.show()

# plt.figure()
# plt.title("beta")
# plt.plot(bets, label="PINN estimate")
# plt.hlines(6.7, 0, len(bets), label="True value", color="tab:green")
# plt.legend()
# plt.xlabel("Training step")
# plt.show()

# plt.figure()
# plt.title("gamma")
# plt.plot(gams, label="PINN estimate")
# plt.hlines(2, 0, len(gams), label="True value", color="tab:green")
# plt.legend()
# plt.xlabel("Training step")
# plt.show()