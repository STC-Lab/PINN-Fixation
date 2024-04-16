
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

def exact_solution(x,t):
    "Defines the analytical solution to the under-damped harmonic oscillator problem above."
    u = 4*torch.exp((-2/3)*(2*3.14159265)**2*t)*torch.sin(2*3.14159265*x)+4*torch.exp((-2/3)*(7*3.14159265)**2*t)*torch.sin(7*3.14159265*x)
    return u

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
    
def plotdataphysics(x_data,t_data,x_physics,t_physics):
    # visualize collocation points for 2D input space (x, t)
    plt.figure()
    plt.scatter(x_data.detach().numpy(), t_data.detach().numpy(),s=4., c='blue', marker='o', label='Data points')
    plt.scatter(x_physics.detach().numpy(), t_physics.detach().numpy(),s=4., c='red', marker='o', label='Physics points')
    plt.title('Samples of the PDE solution y(x,t) for training')
    plt.xlim(0., 1.)
    plt.ylim(0., 1.)
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('t')
    plt.legend('Data points and physics points')
    plt.show()
    plt.show(block=True)

# first, create some noisy observational data
torch.manual_seed(123)
print(f"True value of alpha: {0.66667}")
t_data = torch.rand(25).view(-1,1)
x_data = torch.rand(25).view(-1,1)
X_data,T_data = np.meshgrid(x_data,t_data)
X_data_tensor = torch.tensor(X_data, dtype=torch.float32).view(-1,1)
T_data_tensor = torch.tensor(T_data, dtype=torch.float32).view(-1,1)
U_data = exact_solution(X_data_tensor,T_data_tensor)
U_data_tensor = torch.tensor(U_data, dtype=torch.float32).view(-1,1)

t_physics = torch.linspace(0,1,25).view(-1,1)
x_physics = torch.linspace(0,1,25).view(-1,1)
X_physics,T_physics = np.meshgrid(x_physics,t_physics)
X_physics_tensor = torch.tensor(X_physics, dtype=torch.float32).view(-1,1)
T_physics_tensor = torch.tensor(T_physics, dtype=torch.float32).view(-1,1)


plotdataphysics(X_data_tensor,T_data_tensor,X_physics_tensor,T_physics_tensor)

X_data_tensor.requires_grad = True
T_data_tensor.requires_grad = True
U_data_tensor.requires_grad = True
X_physics_tensor.requires_grad = True
T_physics_tensor.requires_grad = True
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# surf = ax.plot_surface(X_data, T_data, U_data, cmap='viridis', edgecolor='none')
# ax.scatter(X_data,T_data,U_data,color='red', label='Data Points')
# fig.colorbar(surf, ax=ax, label='u(x, t)')
# ax.set_title('Surface Plot of u(x, t)')
# ax.set_xlabel('x')
# ax.set_ylabel('t')
# ax.set_zlabel('u(x, t)')
# plt.show()




torch.manual_seed(123)

# define a neural network to train
pinn = FCN(2,1,32,2)
alpha = torch.nn.Parameter(torch.ones(1, requires_grad=True))
alps = []

# add mu to the optimiser
# TODO: write code here
optimiser = torch.optim.Adam(list(pinn.parameters())+[alpha],lr=1e-3)
writer = SummaryWriter()


for i in range(80001):
    
    optimiser.zero_grad()
    
    # compute each term of the PINN loss function above
    # using the following hyperparameters:
    lambda1 = 1e4
    
    # compute physics loss
    physics_input = [X_physics_tensor,T_physics_tensor]
    physic_output = pinn(physics_input)
    dudt = torch.autograd.grad(physic_output, T_physics_tensor, torch.ones_like(physic_output), create_graph=True)[0]
    dudx = torch.autograd.grad(physic_output, X_physics_tensor, torch.ones_like(physic_output), create_graph=True)[0]
    d2udx2 = torch.autograd.grad(dudx, X_physics_tensor, torch.ones_like(dudt), create_graph=True)[0]
    loss1 = torch.mean((alpha*d2udx2-dudt)**2)
    
    # compute data loss
    # TODO: write code here
    data_input = [X_data_tensor,T_data_tensor]
    data_output = pinn(data_input)
    loss2 = torch.mean((U_data_tensor - data_output)**2)
    
    # backpropagate joint loss, take optimiser step
    loss = loss1 + lambda1*loss2
    loss.backward()
    optimiser.step()
    
    # record mu value
    # TODO: write code here
    alps.append(alpha.item())
    writer.add_scalar('train_loss',loss,i)
    # plot the result as training progresses
    if i % 500 == 0: 
        # u = pinn(t_test).detach()
        # plt.figure(figsize=(6,2.5))
        # plt.scatter(t_obs[:,0], u_obs[:,0], label="Noisy observations", alpha=0.6)
        # plt.plot(t_test[:,0], u[:,0], label="PINN solution", color="tab:green")
        # plt.title(f"Training step {i}")
        # plt.legend()
        # plt.show()
        print(f'epoch: {i}  train loss :{loss}' )
        
plt.figure()
plt.title("alpha")
plt.plot(alps, label="PINN estimate")
plt.hlines(0.666667, 0, len(alps), label="True value", color="tab:green")
plt.legend()
plt.xlabel("Training step")
plt.show()