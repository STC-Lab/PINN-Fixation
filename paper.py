
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
    


torch.manual_seed(4321)
# first, create some noisy observational data
file = './dataset/paper_x10_t0.1.mat'
x,t,u1,u2,u3,u4,u5 = dataprocessing.load_paper(file)
X,T,U1,U2,U3,U4,U5 = dataprocessing.paper_totensor(x,t,u1,u2,u3,u4,u5)
X_test,T_test,U1_test,U2_test,U3_test,U4_test,U5_test = dataprocessing.reshape_paper(X,T,U1,U2,U3,U4,U5)
total_points=len(x[0])*len(t[0])
print('The dataset has',total_points,'points')
Nf =  100 # Nf: Number of collocation points 
X_data_tensor,T_data_tensor,U1_data_tensor,U2_data_tensor,U3_data_tensor,U4_data_tensor,U5_data_tensor = dataprocessing.full_data_paper(total_points,Nf,X_test,T_test,U1_test,U2_test,U3_test,U4_test,U5_test)
# Nf =  int(total_points/2)
# X_data_tensor,T_data_tensor,U1_data_tensor,U2_data_tensor,X_physics_tensor,T_physics_tensor,U1_physics_tensor,U2_physics_tensor = dataprocessing.full_data_matrix(total_points,Nf,X_test,T_test,U1_test,U2_test)

Nf_val =int(Nf*0.1)     #Num of validation set
X_data_tensor_train,X_data_tensor_val,T_data_tensor_train,T_data_tensor_val,U1_data_tensor_train,U1_data_tensor_val,U2_data_tensor_train,U2_data_tensor_val,U3_data_tensor_train,U3_data_tensor_val,U4_data_tensor_train,U4_data_tensor_val,U5_data_tensor_train,U5_data_tensor_val= dataprocessing.split_paper(X_data_tensor,T_data_tensor,U1_data_tensor,U2_data_tensor,U3_data_tensor,U4_data_tensor,U5_data_tensor,Nf_val)


#Add Latin hyper cube sampling
num_samples = 100     # 5000 physics points
parameter_ranges = np.array([[0, 1], [0,1]])
samples = lhs(2, samples=num_samples, criterion='maximin', iterations=1000)
for i in range(2):
    samples[:, i] = samples[:, i] * (parameter_ranges[i, 1] - parameter_ranges[i, 0]) + parameter_ranges[i, 0]
x_samples = samples[:, 0]
t_samples = samples[:, 1]
X_physics,T_physics = np.meshgrid(x_samples,t_samples)
X_physics_tensor = torch.tensor(X_physics, dtype=torch.float32).view(-1,1)
T_physics_tensor = torch.tensor(T_physics, dtype=torch.float32).view(-1,1)


# IC
T_zero = np.zeros(20)
X_ic,T_ic = np.meshgrid(x,T_zero)
X_ic_tensor = torch.tensor(X_ic, dtype=torch.float32).view(-1,1)
T_ic_tensor = torch.tensor(T_ic, dtype=torch.float32).view(-1,1)
#BC
# X_zero = np.zeros(90)
# X_one = np.ones(90)
# X_bc_left,T_bc = np.meshgrid(X_zero,t_samples)
# X_bc_left_tensor = torch.tensor(X_bc_left, dtype=torch.float32).view(-1,1)
# T_bc_tensor = torch.tensor(T_bc, dtype=torch.float32).view(-1,1)
# X_bc_right,T_bc = np.meshgrid(X_one,t_samples)
# X_bc_right_tensor = torch.tensor(X_bc_right, dtype=torch.float32).view(-1,1)



X_data_tensor.requires_grad = True
T_data_tensor.requires_grad = True
U1_data_tensor.requires_grad = True
U2_data_tensor.requires_grad = True
X_physics_tensor.requires_grad = True
T_physics_tensor.requires_grad = True
X_ic_tensor.requires_grad = True
T_ic_tensor.requires_grad = True
X_data_tensor_val.requires_grad = True
T_data_tensor_val.requires_grad = True
# X_bc_left_tensor.requires_grad = True
# X_bc_right_tensor.requires_grad = True

# plotdataphysics(X_data_tensor,T_data_tensor,X_physics_tensor,T_physics_tensor)




# define a neural network to train
pinn = FCN(2,5,64,5)
# pinn = FCN_2output(2,1,32,2)
#pinn = torch.load('C:\yhy\graduation\code\Fixation\model/paper_pretrain_64_5_difflr_10_1_0.0001.pkl')


# alpha = torch.nn.Parameter(torch.randn(2, 2), requires_grad=True)
# mu = torch.nn.Parameter(torch.randn(2,2), requires_grad=True)


alpha1 = torch.nn.Parameter(torch.ones(1, requires_grad=True))
alpha2 = torch.nn.Parameter(torch.ones(1, requires_grad=True))
alpha3 = torch.nn.Parameter(torch.ones(1, requires_grad=True))
alpha4 = torch.nn.Parameter(torch.ones(1, requires_grad=True))
alpha5 = torch.nn.Parameter(torch.ones(1, requires_grad=True))

# alpha1 = torch.nn.Parameter(torch.tensor(2400.0),requires_grad=True)
# alpha2 = torch.nn.Parameter(torch.tensor(1.0),requires_grad=True)
# alpha3 = torch.nn.Parameter(torch.tensor(400.0),requires_grad=True)
# alpha4 = torch.nn.Parameter(torch.tensor(1.0),requires_grad=True)
# alpha5 = torch.nn.Parameter(torch.tensor(200.0),requires_grad=True)



alps1 = []
alps2 = []
alps3 = []
alps4 = []
alps5 = []
physicsloss = []
dataloss = []
totalloss = []
valloss = []

# add mu to the optimiser
# TODO: write code here
# optimiser = torch.optim.Adam(list(pinn.parameters())+[alpha1]+[alpha2]+[alpha3]+[alpha4]+[alpha5],lr=0.001)
optimiser1 = torch.optim.Adam(list(pinn.parameters())+[alpha2]+[alpha4],lr=0.001)
optimiser2 = torch.optim.Adam([alpha1],lr=0.1)
optimiser3 = torch.optim.Adam([alpha3]+[alpha5],lr=0.01)

# optimiser1 = torch.optim.Adam(list(pinn.parameters())+[alpha2]+[alpha4],lr=0.01)
# optimiser2 = torch.optim.Adam([alpha1]+[alpha3]+[alpha5],lr=0.1)
writer = SummaryWriter()

# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', patience=2000, factor=0.1)

theta = 0.001
loss = 1
i = 0

time_start = time.time()

try:
    while loss > theta:
    # for i in range(70001):
        
        optimiser1.zero_grad()
        optimiser2.zero_grad()
        optimiser3.zero_grad()
        # compute each term of the PINN loss function above
        # using the following hyperparameters:
        lambda1 = 1e4
        
        # compute physics loss
        physics_input = [X_physics_tensor,T_physics_tensor]
        physic_output = pinn(physics_input)
        physic_output1 = physic_output[:, 0].view(-1, 1)
        physic_output2 = physic_output[:, 1].view(-1, 1)
        physic_output3 = physic_output[:, 2].view(-1, 1)
        physic_output4 = physic_output[:, 3].view(-1, 1)
        physic_output5 = physic_output[:, 4].view(-1, 1)
        du1dt = torch.autograd.grad(physic_output1, T_physics_tensor, torch.ones_like(physic_output1), create_graph=True)[0]
        du2dt = torch.autograd.grad(physic_output2, T_physics_tensor, torch.ones_like(physic_output2), create_graph=True)[0]
        du3dt = torch.autograd.grad(physic_output3, T_physics_tensor, torch.ones_like(physic_output3), create_graph=True)[0]
        du4dt = torch.autograd.grad(physic_output4, T_physics_tensor, torch.ones_like(physic_output4), create_graph=True)[0]
        du5dt = torch.autograd.grad(physic_output5, T_physics_tensor, torch.ones_like(physic_output5), create_graph=True)[0]
        du1dx = torch.autograd.grad(physic_output1, X_physics_tensor, torch.ones_like(physic_output1), create_graph=True)[0]
        du2dx = torch.autograd.grad(physic_output2, X_physics_tensor, torch.ones_like(physic_output2), create_graph=True)[0]
        du3dx = torch.autograd.grad(physic_output3, X_physics_tensor, torch.ones_like(physic_output3), create_graph=True)[0]
        du4dx = torch.autograd.grad(physic_output4, X_physics_tensor, torch.ones_like(physic_output4), create_graph=True)[0]
        du5dx = torch.autograd.grad(physic_output5, X_physics_tensor, torch.ones_like(physic_output5), create_graph=True)[0]
        d2u1dx2 = torch.autograd.grad(du1dx, X_physics_tensor, torch.ones_like(du1dx), create_graph=True)[0]
        d2u2dx2 = torch.autograd.grad(du2dx, X_physics_tensor, torch.ones_like(du2dx), create_graph=True)[0]
        d2u3dx2 = torch.autograd.grad(du3dx, X_physics_tensor, torch.ones_like(du3dx), create_graph=True)[0]
        d2u4dx2 = torch.autograd.grad(du4dx, X_physics_tensor, torch.ones_like(du4dx), create_graph=True)[0]
        d2u5dx2 = torch.autograd.grad(du5dx, X_physics_tensor, torch.ones_like(du5dx), create_graph=True)[0]


        #Physics loss
        loss1 = torch.mean((alpha1*d2u1dx2-du1dt)**2+(alpha2*d2u2dx2-du2dt)**2+(alpha3*d2u3dx2-du3dt)**2+(alpha4*d2u4dx2-du4dt)**2+(alpha5*d2u5dx2-du5dt)**2)
        # loss1 = torch.mean((alpha[0][0]*d2u1dx2+alpha[0][1]*d2u2dx2+mu[0][0]*physic_output1+mu[0][1]*physic_output2-du1dt)**2+(alpha[1][0]*d2u1dx2+alpha[1][1]*d2u2dx2+mu[1][0]*physic_output1+mu[1][1]*physic_output2-du2dt)**2)
        physicsloss.append(loss1.item())
        writer.add_scalar('loss1',loss1,i)

        

        # compute data loss
        # TODO: write code here
        data_input = [X_data_tensor_train,T_data_tensor_train]
        data_output = pinn(data_input)
        data_output1 = data_output[:, 0].view(-1, 1)
        data_output2 = data_output[:, 1].view(-1, 1)
        data_output3 = data_output[:, 2].view(-1, 1)
        data_output4 = data_output[:, 3].view(-1, 1)
        data_output5 = data_output[:, 4].view(-1, 1)
        loss2 = torch.mean((U1_data_tensor_train - data_output1)**2+(U2_data_tensor_train-data_output2)**2+(U3_data_tensor_train-data_output3)**2+(U4_data_tensor_train-data_output4)**2+(U5_data_tensor_train-data_output5)**2)
        # loss2 = torch.mean((U1_data_tensor - data_output1)**2+(U2_data_tensor-data_output2)**2)
        dataloss.append(loss2.item())
        writer.add_scalar('loss2',loss2,i)



        #Initial Condition
        # ic_input = [X_ic_tensor,T_ic_tensor]
        # ic_output = pinn(ic_input)
        # ic_output1 = ic_output[:, 0].view(-1, 1)
        # ic_output2 = ic_output[:, 1].view(-1, 1)
        # ic_output3 = ic_output[:, 2].view(-1, 1)
        # ic_output4 = ic_output[:, 3].view(-1, 1)
        # ic_output5 = ic_output[:, 4].view(-1, 1)
        # loss3 = torch.mean((ic_output1-((4352*X_ic_tensor**2)/2585
        #                                 -(8704*X_ic_tensor)/2585+15760/517))**2+(ic_output2
        #                                                                           -144/5)**2+(ic_output3
        #                                                                                         -(-492*X_ic_tensor**3/5+(738*X_ic_tensor**2)/5+144/5))**2+(ic_output4-78)**2+(ic_output5-(78-(13*X_ic_tensor**2)/51))**2)
        # writer.add_scalar('loss3',loss3,i)





        # backpropagate joint loss, take optimiser step
        # loss1 = loss11 + lambda1*loss21
        # loss2 = loss12 + lambda1*loss22
        # loss = loss1 + lambda1*(loss2+loss3+loss4)
        # constraint = 0.01/(alpha1+1e-8)+0.01/(alpha2+1e-8)+0.01/(alpha3+1e-8)+0.01/(alpha4+1e-8)+0.01/(alpha5+1e-8)
        # constraint = abs(alpha1-2500)+abs(alpha3-500)+abs(alpha5-250)
        loss = loss1+lambda1*loss2
        totalloss.append(loss.item())
        loss.backward()
        optimiser1.step()
        optimiser2.step()
        optimiser3.step()
        # optimiser2.step()
        # scheduler.step(loss)
        
        # record mu value
        # TODO: write code here
        alps1.append(alpha1.item())
        alps2.append(alpha2.item())
        alps3.append(alpha3.item())
        alps4.append(alpha4.item())
        alps5.append(alpha5.item())


        physics_input_val = [X_data_tensor_val,T_data_tensor_val]
        physic_output_val = pinn(physics_input_val)
        physic_output1_val = physic_output_val[:, 0].view(-1, 1)
        physic_output2_val = physic_output_val[:, 1].view(-1, 1)
        physic_output3_val = physic_output_val[:, 2].view(-1, 1)
        physic_output4_val = physic_output_val[:, 3].view(-1, 1)
        physic_output5_val = physic_output_val[:, 4].view(-1, 1)
        du1dt_val = torch.autograd.grad(physic_output1_val, T_data_tensor_val, torch.ones_like(physic_output1_val), create_graph=True)[0]
        du2dt_val = torch.autograd.grad(physic_output2_val, T_data_tensor_val, torch.ones_like(physic_output2_val), create_graph=True)[0]
        du3dt_val = torch.autograd.grad(physic_output3_val, T_data_tensor_val, torch.ones_like(physic_output3_val), create_graph=True)[0]
        du4dt_val = torch.autograd.grad(physic_output4_val, T_data_tensor_val, torch.ones_like(physic_output4_val), create_graph=True)[0]
        du5dt_val = torch.autograd.grad(physic_output5_val, T_data_tensor_val, torch.ones_like(physic_output5_val), create_graph=True)[0]
        du1dx_val = torch.autograd.grad(physic_output1_val, X_data_tensor_val, torch.ones_like(physic_output1_val), create_graph=True)[0]
        du2dx_val = torch.autograd.grad(physic_output2_val, X_data_tensor_val, torch.ones_like(physic_output2_val), create_graph=True)[0]
        du3dx_val = torch.autograd.grad(physic_output3_val, X_data_tensor_val, torch.ones_like(physic_output3_val), create_graph=True)[0]
        du4dx_val = torch.autograd.grad(physic_output4_val, X_data_tensor_val, torch.ones_like(physic_output4_val), create_graph=True)[0]
        du5dx_val = torch.autograd.grad(physic_output5_val, X_data_tensor_val, torch.ones_like(physic_output5_val), create_graph=True)[0]
        d2u1dx2_val = torch.autograd.grad(du1dx_val, X_data_tensor_val, torch.ones_like(du1dx_val), create_graph=True)[0]
        d2u2dx2_val = torch.autograd.grad(du2dx_val, X_data_tensor_val, torch.ones_like(du2dx_val), create_graph=True)[0]
        d2u3dx2_val = torch.autograd.grad(du3dx_val, X_data_tensor_val, torch.ones_like(du3dx_val), create_graph=True)[0]
        d2u4dx2_val = torch.autograd.grad(du4dx_val, X_data_tensor_val, torch.ones_like(du4dx_val), create_graph=True)[0]
        d2u5dx2_val = torch.autograd.grad(du5dx_val, X_data_tensor_val, torch.ones_like(du5dx_val), create_graph=True)[0]


        #Physics loss
        loss1_val = torch.mean((alpha1*d2u1dx2_val-du1dt_val)**2+(alpha2*d2u2dx2_val-du2dt_val)**2+(alpha3*d2u3dx2_val-du3dt_val)**2+(alpha4*d2u4dx2_val-du4dt_val)**2+(alpha5*d2u5dx2_val-du5dt_val)**2)
        # compute data loss
        # TODO: write code here
        data_input_val = [X_data_tensor_val,T_data_tensor_val]
        data_output_val = pinn(data_input_val)
        data_output1_val = data_output_val[:, 0].view(-1, 1)
        data_output2_val = data_output_val[:, 1].view(-1, 1)
        data_output3_val = data_output_val[:, 2].view(-1, 1)
        data_output4_val = data_output_val[:, 3].view(-1, 1)
        data_output5_val= data_output_val[:, 4].view(-1, 1)
        loss2_val = torch.mean((U1_data_tensor_val - data_output1_val)**2+(U2_data_tensor_val-data_output2_val)**2+(U3_data_tensor_val-data_output3_val)**2+(U4_data_tensor_val-data_output4_val)**2+(U5_data_tensor_val-data_output5_val)**2)
        # backpropagate joint loss, take optimiser step
        loss_val = loss1_val + lambda1*loss2_val
        writer.add_scalar('val_loss',loss_val,i)
        valloss.append(loss_val.item())



        if i % 500 == 0: 

             print(f'epoch: {i}  train loss :{loss}, valloss:{loss_val} alpha: {alpha1,alpha2,alpha3,alpha4,alpha5}' )
        i = i+1
except KeyboardInterrupt:
    print("Interrupted training loop.")



data = {'alpha1':alps1,
         'alpha2':alps2,
         'alpha3':alps3,
         'alpha4':alps4,
         'alpha5':alps5,   
         'physicsloss':physicsloss,
         'dataloss':dataloss,
         'totalloss':totalloss,
         'valloss':valloss,
}

with open('paper_x10_t0.1_0.001_0.1_0.01.json',"w") as f:
    json.dump(data,f,indent=4)

torch.save(pinn,"./model/paper_x10_t0.1_0.001_0.1_0.01.pkl")
time_end = time.time()
time_sum = time_end - time_start
print('训练时间 {:.0f}分 {:.0f}秒'.format(time_sum // 60, time_sum % 60))
        

# plt.figure()
# plt.title("alpha_00")
# plt.plot(alps1, label="PINN estimate")
# plt.hlines(3, 0, len(alps1), label="True value", color="tab:green")
# plt.legend()
# plt.xlabel("Training step")
# plt.show()

# plt.figure()
# plt.title("alpha_01")
# plt.plot(alps2, label="PINN estimate")
# plt.hlines(1, 0, len(alps2), label="True value", color="tab:green")
# plt.legend()
# plt.xlabel("Training step")
# plt.show()

# plt.figure()
# plt.title("alpha_10")
# plt.plot(alps3, label="PINN estimate")
# plt.hlines(2, 0, len(alps3), label="True value", color="tab:green")
# plt.legend()
# plt.xlabel("Training step")
# plt.show()

# plt.figure()
# plt.title("alpha_11")
# plt.plot(alps4, label="PINN estimate")
# plt.hlines(1.5, 0, len(alps4), label="True value", color="tab:green")
# plt.legend()
# plt.xlabel("Training step")
# plt.show()

# plt.figure()
# plt.title("mu_00")
# plt.plot(mus1, label="PINN estimate")
# plt.hlines(2, 0, len(mus1), label="True value", color="tab:green")
# plt.legend()
# plt.xlabel("Training step")
# plt.show()

# plt.figure()
# plt.title("mu_01")
# plt.plot(mus2, label="PINN estimate")
# plt.hlines(1.8, 0, len(mus2), label="True value", color="tab:green")
# plt.legend()
# plt.xlabel("Training step")
# plt.show()

# plt.figure()
# plt.title("mu_10")
# plt.plot(mus3, label="PINN estimate")
# plt.hlines(1.2, 0, len(mus3), label="True value", color="tab:green")
# plt.legend()
# plt.xlabel("Training step")
# plt.show()

# plt.figure()
# plt.title("mu_11")
# plt.plot(mus4, label="PINN estimate")
# plt.hlines(-2.5, 0, len(mus4), label="True value", color="tab:green")
# plt.legend()
# plt.xlabel("Training step")
# plt.show()