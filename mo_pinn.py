
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from scipy.io import loadmat
import data_fixation
import time
import numpy as np
from pyDOE import lhs

device = 'cpu'

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
    

torch.manual_seed(1234)
# first, create some noisy observational data
file = './dataset/fixation_1.mat'
x,t,te,mo,Dc = data_fixation.load_data(file)
X,T,Te,Mo = data_fixation.totensor(x,t,te,mo)
X_test,T_test,Te_test,Mo_test = data_fixation.reshape_data(X,T,Te,Mo)
total_points=len(x[0])*len(t[0])
print('The dataset has',total_points,'points')
Nf =  4356 # Nf: Number of collocation points 
X_data_tensor,T_data_tensor,Te_data_tensor,Mo_data_tensor = data_fixation.select_random_data(total_points,Nf,X_test,T_test,Te_test,Mo_test)
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
num_samples = 66
parameter_ranges = np.array([[0, 2], [0, 20]])
samples = lhs(2, samples=num_samples, criterion='maximin', iterations=1000)
for i in range(2):
    samples[:, i] = samples[:, i] * (parameter_ranges[i, 1] - parameter_ranges[i, 0]) + parameter_ranges[i, 0]
x_samples = samples[:, 0]
t_samples = samples[:, 1]
X_physics,T_physics = np.meshgrid(x_samples,t_samples)
X_physics_tensor = torch.tensor(X_physics, dtype=torch.float32).view(-1,1)
T_physics_tensor = torch.tensor(T_physics, dtype=torch.float32).view(-1,1)


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
# X_bc_right,T_bc = np.meshgrid(X_one,t_samples)
# X_bc_right_tensor = torch.tensor(X_bc_right, dtype=torch.float32).view(-1,1)



X_data_tensor.requires_grad = True
T_data_tensor.requires_grad = True
Te_data_tensor.requires_grad = True
Mo_data_tensor.requires_grad = True
X_physics_tensor.requires_grad = True
T_physics_tensor.requires_grad = True
# X_bc_left_tensor.requires_grad = True
# X_bc_right_tensor.requires_grad = True

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

index = 2                                  #   Num of index of the material
alpha = [torch.nn.Parameter(torch.randn(2, 2).to(device), requires_grad=True) for _ in range(index)] 
# beta = torch.nn.Parameter(torch.randn(2), requires_grad=True)
# gamma = torch.nn.Parameter(torch.randn(2), requires_grad=True)

# alpha1 = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
# beta1 = torch.nn.Parameter(torch.tensor(1.0),requires_grad=True)
# gamma1 = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
# alpha2 = torch.nn.Parameter(torch.tensor(1.0),requires_grad=True)
# beta2 = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
# gamma2 = torch.nn.Parameter(torch.tensor(1.0),requires_grad=True)

alps11 = []
alps12 = []
alps13 = []
alps14 =[]
alps21 = []
alps22 = []
alps23 = []
alps24 =[]


# add mu to the optimiser
# TODO: write code here
optimiser = torch.optim.Adam(list(pinn.parameters())+alpha,lr=0.01)
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
        physic_output_Te = physic_output[:, 0].view(-1, 1)
        physic_output_Mo = physic_output[:, 1].view(-1, 1)

        dTedt = torch.autograd.grad(physic_output_Te, T_physics_tensor, torch.ones_like(physic_output_Te), create_graph=True)[0]
        dModt = torch.autograd.grad(physic_output_Mo, T_physics_tensor, torch.ones_like(physic_output_Mo), create_graph=True)[0]
        dTedx = torch.autograd.grad(physic_output_Te, X_physics_tensor, torch.ones_like(physic_output_Te), create_graph=True)[0]
        dModx = torch.autograd.grad(physic_output_Mo, X_physics_tensor, torch.ones_like(physic_output_Mo), create_graph=True)[0]
        d2Tedx2 = torch.autograd.grad(dTedx, X_physics_tensor, torch.ones_like(dTedx), create_graph=True)[0]
        d2Modx2 = torch.autograd.grad(dModx, X_physics_tensor, torch.ones_like(dModx), create_graph=True)[0]
        loss1 = (alpha[0][0][0]*d2Tedx2+alpha[0][0][1]*d2Modx2-dTedt)**2+(alpha[0][1][0]*d2Tedx2+alpha[0][1][1]*d2Modx2-dTedt)**2
        if index > 1:
            for j in range(1,index):
                # loss1 = loss1+ (alpha[i][2*i][2*i]*d2Tedx2+alpha[i][2*i][2*i+1]*d2Modx2-dTedt)**2+(alpha[i][2*i+1][2*i]*d2Tedx2+alpha[i][2*i+1][2*i+1]*d2Modx2-dTedt)**2
                loss1 = loss1+ (alpha[j][0][0]*d2Tedx2+alpha[j][0][1]*d2Modx2-dTedt)**2+(alpha[j][1][0]*d2Tedx2+alpha[j][1][1]*d2Modx2-dTedt)**2
        # loss1 = torch.mean((alpha1*d2u1dx2+beta1*du1dx+gamma1*physic_output1-du1dt)**2+(alpha2*d2u2dx2+beta2*du2dx+gamma2*physic_output2-du2dt)**2)
        # loss1 = torch.mean((alpha[0]*d2u1dx2+gamma[0]*physic_output1-du1dt)**2+(alpha[1]*d2u2dx2+gamma[1]*physic_output1-du2dt)**2)
        # loss11 = torch.mean((alpha1*d2u1dx2+beta1*du1dx+gamma1*physic_output1-du1dt)**2)
        # loss12 = torch.mean((alpha2*d2u2dx2+beta2*du2dx+gamma2*physic_output2-du2dt)**2)
        loss1 = torch.mean(loss1)
        writer.add_scalar('loss1',loss1,i)
        # writer.add_scalar('loss12',loss12,i)
        

        # compute data loss
        # TODO: write code here
        data_input = [X_data_tensor,T_data_tensor]
        data_output = pinn(data_input)
        data_output_Te = data_output[:, 0].view(-1, 1)
        data_output_Mo = data_output[:, 1].view(-1, 1)
        loss2 = torch.mean((Te_data_tensor - data_output_Te)**2+(Mo_data_tensor-data_output_Mo)**2)
        # loss21 = torch.mean((U1_data_tensor - data_output1)**2)
        # loss22 = torch.mean((U2_data_tensor-data_output2)**2)
        writer.add_scalar('loss2',loss2,i)
        # writer.add_scalar('loss22',loss22,i)


        # #Initial Condition
        # ic_input = [X_ic_tensor,T_ic_tensor]
        # ic_output = pinn(ic_input)
        # ic_output1 = ic_output[:, 0].view(-1, 1)
        # ic_output2 = ic_output[:, 1].view(-1, 1)
        # loss3 = torch.mean((ic_output1-np.sin(0.5*np.pi*X_ic_tensor))**2+(ic_output2-np.sin(0.5*np.pi*X_ic_tensor))**2)
        # writer.add_scalar('loss3',loss3,i)


        # #Boundry Condition
        # bc_input_left = [X_bc_left_tensor,T_bc_tensor]
        # bc_output_left = pinn(bc_input_left)
        # bc_left_output1 = bc_output_left[:, 0].view(-1, 1)
        # bc_left_output2 = bc_output_left[:, 1].view(-1, 1)
        # bc_input_right = [X_bc_right_tensor,T_bc_tensor]
        # bc_output_right = pinn(bc_input_right)
        # bc_right_output1 = bc_output_right[:, 0].view(-1, 1)
        # bc_right_output2 = bc_output_right[:, 1].view(-1, 1)
        # du2dx_bc = torch.autograd.grad(bc_left_output1, X_bc_left_tensor, torch.ones_like(bc_left_output1), create_graph=True)[0]
        # du1dx_bc = torch.autograd.grad(bc_right_output2, X_bc_right_tensor, torch.ones_like(bc_right_output2), create_graph=True)[0]

        # loss4 = torch.mean((bc_left_output2)**2)+torch.mean((bc_right_output1-1)**2)+torch.mean((du2dx_bc)**2)+torch.mean((du1dx_bc)**2)
        # writer.add_scalar('loss4',loss4,i)

        


        # backpropagate joint loss, take optimiser step
        # loss1 = loss11 + lambda1*loss21
        # loss2 = loss12 + lambda1*loss22
        loss = loss1 + lambda1*loss2
        loss.backward()
        optimiser.step()
        
        # record mu value
        # TODO: write code here
        alps11.append(alpha[0][0][0].item())
        alps12.append(alpha[0][0][1].item())
        alps13.append(alpha[0][1][0].item())
        alps14.append(alpha[0][1][1].item())
        alps21.append(alpha[1][0][0].item())
        alps22.append(alpha[1][0][1].item())
        alps23.append(alpha[1][1][0].item())
        alps24.append(alpha[1][1][1].item())
        writer.add_scalar('train_loss',loss,i)
        writer.add_scalar('alpha11',alpha[0][0][0],i)
        writer.add_scalar('alpha12',alpha[0][0][1],i)
        writer.add_scalar('alpha13',alpha[0][1][0],i)
        writer.add_scalar('alpha14',alpha[0][1][1],i)
        writer.add_scalar('alpha21',alpha[1][0][0],i)
        writer.add_scalar('alpha22',alpha[1][0][1],i)
        writer.add_scalar('alpha23',alpha[1][1][0],i)
        writer.add_scalar('alpha24',alpha[1][1][1],i)

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
             print(f'epoch: {i}  train loss :{loss}, alpha: {alpha[0],alpha[1]}' )
        i = i+1
except KeyboardInterrupt:
    print("Interrupted training loop.")


        
torch.save(pinn,"./model/13560502.pkl.")
time_end = time.time()
time_sum = time_end - time_start
print('训练时间 {:.0f}分 {:.0f}秒'.format(time_sum // 60, time_sum % 60))
        
plt.figure()
plt.title("alpha_11")
plt.plot(alps11, label="PINN estimate")
plt.hlines(5, 0, len(alps11), label="True value", color="tab:green")
plt.legend()
plt.xlabel("Training step")
plt.show()

plt.figure()
plt.title("alpha_12")
plt.plot(alps12, label="PINN estimate")
plt.hlines(0.1, 0, len(alps12), label="True value", color="tab:green")
plt.legend()
plt.xlabel("Training step")
plt.show()

plt.figure()
plt.title("alpha_13")
plt.plot(alps13, label="PINN estimate")
plt.hlines(0.2, 0, len(alps13), label="True value", color="tab:green")
plt.legend()
plt.xlabel("Training step")
plt.show()

plt.figure()
plt.title("alpha_14")
plt.plot(alps14, label="PINN estimate")
plt.hlines(3, 0, len(alps14), label="True value", color="tab:green")
plt.legend()
plt.xlabel("Training step")
plt.show()

plt.figure()
plt.title("alpha_21")
plt.plot(alps21, label="PINN estimate")
plt.hlines(3, 0, len(alps21), label="True value", color="tab:green")
plt.legend()
plt.xlabel("Training step")
plt.show()

plt.figure()
plt.title("alpha_22")
plt.plot(alps22, label="PINN estimate")
plt.hlines(0.2, 0, len(alps22), label="True value", color="tab:green")
plt.legend()
plt.xlabel("Training step")
plt.show()

plt.figure()
plt.title("alpha_23")
plt.plot(alps23, label="PINN estimate")
plt.hlines(0.5, 0, len(alps23), label="True value", color="tab:green")
plt.legend()
plt.xlabel("Training step")
plt.show()

plt.figure()
plt.title("alpha_24")
plt.plot(alps24, label="PINN estimate")
plt.hlines(4, 0, len(alps24), label="True value", color="tab:green")
plt.legend()
plt.xlabel("Training step")
plt.show()