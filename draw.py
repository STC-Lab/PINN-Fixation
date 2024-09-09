import torch
import dataprocessing
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
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

def plot3D(X, T, y):
    X = X.detach().numpy()
    T = T.detach().numpy()
    y = y.detach().numpy()

    #     2D
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    cm = ax1.contourf(T, X, y, 20,cmap="viridis")
    fig.colorbar(cm, ax=ax1) # Add a colorbar to a plot
    ax1.set_title('u(x,t)')
    ax1.set_xlabel('t')
    ax1.set_ylabel('x')
    ax1.set_aspect('equal')
        #     3D
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(T, X, y,cmap="viridis")
    ax2.set_xlabel('t')
    ax2.set_ylabel('x')
    ax2.set_zlabel('u(x,t)')
    fig.tight_layout()
    plt.show()


#17260426.pkl  3matrix_2-5_3-6_4-7.mat
#16420430.pkl  3.5-5.5_6.7-0_8.5-0.mat
#18430430.pkl  3.5-5.5_6.7-4.7_8.5-0.mat
#12010502.pkl  3.5-5.5_8.5-0.mat
#18010502.pkl  3.5-5.5_8.5-12.5.mat
model = torch.load('E:\yhy_files\graduation\code\Fixation\model1111/paper_32_3_difflr_0.1_0.01.pkl')       
file = './dataset/paper_test2.mat'



x,t= dataprocessing.load_matrix_test(file)
X,T = dataprocessing.matrix_totensor_test(x,t)
X_test,T_test= dataprocessing.reshape_matrix_test(X,T)
# total_points=len(x[0])*len(t[0])
# print('The dataset has',total_points,'points')
# x,t,u = dataprocessing.load_data(file)
# X,T,U = dataprocessing.totensor(x,t,u)
# X_test,T_test,U_test = dataprocessing.reshape_data(X,T,U)
# total_points=len(x[0])*len(t[0])
# print('The dataset has',total_points,'points')



# input = [X_test,T_test]
# output = model(input)
# usol = output.view(-1, 1).flatten()
# usol = usol.resize(100,128)

# data = {
#     'u': usol.detach().numpy().tolist(),
#     'x':x.tolist(),
#     't':t.tolist(),}


# x,t,u1,u2 = dataprocessing.load_matrix(file)
# X,T,U1,U2 = dataprocessing.matrix_totensor(x,t,u1,u2)
# X_test,T_test,U1_test, U2_test= dataprocessing.reshape_matrix(X,T,U1,U2)
# total_points=len(x[0])*len(t[0])
# print('The dataset has',total_points,'points')


# input = [X_test,T_test]
# output = model(input)
# u = output.view(-1, 1).flatten()
# data_output1 = output[:, 0].view(-1, 1)
# data_output2 = output[:, 1].view(-1, 1)
#loss2 = torch.mean((U1 - data_output1)**2+(U2-data_output2)**2)

# du1dt = torch.autograd.grad(data_output1, T, torch.ones_like(data_output1), create_graph=True)[0]
# du2dt = torch.autograd.grad(data_output2, T, torch.ones_like(data_output2), create_graph=True)[0]
# du1dx = torch.autograd.grad(data_output1, X, torch.ones_like(data_output1), create_graph=True)[0]
# du2dx = torch.autograd.grad(data_output2, X, torch.ones_like(data_output2), create_graph=True)[0]
# d2u1dx2 = torch.autograd.grad(du1dx, X, torch.ones_like(du1dx), create_graph=True)[0]
# d2u2dx2 = torch.autograd.grad(du2dx, X, torch.ones_like(du2dx), create_graph=True)[0]


        #Physics loss
#loss1 = torch.mean((3.0772*d2u1dx2+(-1.0841)*d2u2dx2+2.0493*data_output1+1.7162*data_output2-du1dt)**2+(2.6423*d2u1dx2+1.4935*d2u2dx2+1.1753*data_output1+(-2.5480)*data_output2-du2dt)**2)

# u1 = output[:, 0].view(-1, 1).flatten()
# u2 = output[:, 1].view(-1, 1).flatten()
# u1 = u1.resize(50,50)
# u2 = u2.resize(50,50)

# data = {
#     'u1': u1.detach().numpy().tolist(),
#     'u2': u2.detach().numpy().tolist(),
#     'x':x.tolist(),
#     't':t.tolist(),}



# x,t,u1,u2,u3,u4,u5 = dataprocessing.load_paper(file)
# X,T,U1,U2,U3,U4,U5 = dataprocessing.paper_totensor(x,t,u1,u2,u3,u4,u5)
# X_test,T_test,U1_test,U2_test,U3_test,U4_test,U5_test = dataprocessing.reshape_paper(X,T,U1,U2,U3,U4,U5)
# total_points=len(x[0])*len(t[0])
# print('The dataset has',total_points,'points')


input = [X_test,T_test]
output = model(input)
u1 = output[:, 0].view(-1, 1).flatten()
u2 = output[:, 1].view(-1, 1).flatten()
u3 = output[:, 2].view(-1, 1).flatten()
u4 = output[:, 3].view(-1, 1).flatten()
u5 = output[:, 4].view(-1, 1).flatten()
u1 = u1.resize(50,100)
u2 = u2.resize(50,100)
u3 = u3.resize(50,100)
u4 = u4.resize(50,100)
u5 = u5.resize(50,100)

data = {
    'u1': u1.detach().numpy().tolist(),
    'u2': u2.detach().numpy().tolist(),
    'u3': u3.detach().numpy().tolist(),
    'u4': u4.detach().numpy().tolist(),
    'u5': u5.detach().numpy().tolist(),
    'x':x.tolist(),
    't':t.tolist(),}






with open('paper_draw.json',"w") as f:
    json.dump(data,f,indent=4)
# np.savetxt('u1.json', u1.detach().numpy(), delimiter=',')
# np.savetxt('u2.csv', u2.detach().numpy(), delimiter=',')
# plot3D(X,T,u1)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# surf = ax.plot_surface(X.detach().numpy(), T.detach().numpy(), u1.detach().numpy(), cmap='viridis', edgecolor='none')
# fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
# ax.set_title('u_1(x, t)')
# ax.set_xlabel('Distance x')
# ax.set_ylabel('Time t')
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# surf = ax.plot_surface(X.detach().numpy(), T.detach().numpy(), u2.detach().numpy(), cmap='viridis', edgecolor='none')
# fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
# ax.set_title('u_2(x, t)')
# ax.set_xlabel('Distance x')
# ax.set_ylabel('Time t')
# plt.show()






