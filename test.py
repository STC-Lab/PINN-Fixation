import torch
import dataprocessing
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


#Test the trained model
#Visualize the prediction results
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



model = torch.load('E:\yhy_files\graduation\code\PINN1.0\model/15460424.pkl')

file = './dataset/3matrix_5-5_6-6_7-7.mat'
x,t,u1,u2 = dataprocessing.load_matrix(file)
X,T,U1,U2 = dataprocessing.matrix_totensor(x,t,u1,u2)
X_test,T_test,U1_test,U2_test = dataprocessing.reshape_matrix(X,T,U1,U2)
total_points=len(x[0])*len(t[0])
print('The dataset has',total_points,'points')

input = [X_test,T_test]
output = model(input)
u1 = output[:, 0].view(-1, 1).flatten()
u2 = output[:, 1].view(-1, 1).flatten()
u1 = u1.resize(128,64)
u2 = u2.resize(128,64)

# plot3D(X,T,u1)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X.detach().numpy(), T.detach().numpy(), u1.detach().numpy(), cmap='viridis', edgecolor='none')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
ax.set_title('u_1(x, t)')
ax.set_xlabel('Distance x')
ax.set_ylabel('Time t')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X.detach().numpy(), T.detach().numpy(), u2.detach().numpy(), cmap='viridis', edgecolor='none')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
ax.set_title('u_2(x, t)')
ax.set_xlabel('Distance x')
ax.set_ylabel('Time t')
plt.show()






