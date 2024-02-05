import torch
import torch.nn as nn                    
import numpy as np
# data imports
from scipy.io import loadmat
# plotting imports
import matplotlib.pyplot as plt
import dataprocessing
import visualization
import network
from torch.autograd import Variable
from train import Trainer


#Set default dtype to float32
torch.set_default_dtype(torch.float)
#PyTorch random number generator
torch.manual_seed(1234)
# Random number generators in other libraries
np.random.seed(1234)
# Device configuration
device = torch.device('cpu')

file = './dataset/dataset_example123_2.mat'
x,t,u = dataprocessing.load_data(file)
X,T,U = dataprocessing.totensor(x,t,u)
#visualization.plot3D(X,T,U)       #Visualize the dataset

X_test,T_test,U_test = dataprocessing.reshape_data(X,T,U)      #reshape data to 1D array

total_points=len(x[0])*len(t[0])
Nf =  3 # Nf: Number of collocation points 
x_train,t_train,u_train = dataprocessing.select_data(total_points,Nf,X_test,T_test,U_test) # Select data. Obtain random points of our PDE measurements y(x,t)
#visualization.plotdata(x_train,t_train)   # visualize collocation points for 2D input space (x, t)

# turn on gradients for PINN
x_train.requires_grad=True
t_train.requires_grad=True
train_data = dataprocessing.DictDataset({'x': x_train, 't':t_train, 'u':u_train}, name='train')    # Training dataset
#test_data = dataprocessing.DictDataset({'x': X_test, 't':T_test, 'u':U_test}, name='test')    # Test dataset


# torch dataloaders
#batch_size = x_train.shape[0]  # full batch training
batch_size = 3
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                           collate_fn=train_data.collate_fn,
                                           shuffle=False)
#test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
#                                         collate_fn=test_data.collate_fn,
#                                         shuffle=False)

#net = network.Net(n_input=2,hidden_layer=[4,4],n_output=1,activate_func=nn.Tanh)

net = network.Network(input_size=2,output_size=1,hsizes=[4],nonlin=nn.Tanh)
optimizer = torch.optim.AdamW(net.parameters(), lr=0.001)
epochs = 5
trainer = Trainer(
    net.to(device),
    train_loader,
    optimizer=optimizer,
    epochs=epochs,
    epoch_verbose=200,
)           #  Neuromancer trainer



best_model = trainer.train()