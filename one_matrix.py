
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from scipy.io import loadmat
import dataprocessing


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
    plt.xlim(0., 2.)
    plt.ylim(0., 1.)
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('t')
    plt.legend('Data points and physics points')
    plt.show()
    plt.show(block=True)


torch.manual_seed(1234)
