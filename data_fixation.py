from scipy.io import loadmat
import torch
import torch.nn as nn  
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from sklearn.model_selection import train_test_split
import pyDOE 


def load_data(file):
    data = loadmat(file) 
    fixation = data['fixation']
    x = fixation['x'][0][0]                                   # space:      256 points between -1 and 1 [256x1]
    t = fixation['t'][0][0]                                    # time:       100 time points between 0 and 1 [100x1] 
    T = fixation['T'][0][0] 
    M = fixation['M'][0][0] 
    Dc = fixation['Dc'][0][0] 
    return x,t,T,M,Dc                           # velocitu:   PDE solution [256x100] 


def totensor(x,t,te,mo):
    X, T = np.meshgrid(x,t)                         # makes 2 arrays X and T such that u(X[i],T[j])=usol[i][j] are a tuple
    X = torch.tensor(X.T).float()
    T = torch.tensor(T.T).float()
    Te = torch.tensor(te).float()
    Mo = torch.tensor(mo).float()
    return X,T,Te,Mo


def reshape_data(x,t,te,mo):
    x_test = x.reshape(-1,1)
    t_test = t.reshape(-1,1)
    te_test = te.reshape(-1,1)
    mo_test = mo.reshape(-1,1)
    return x_test,t_test,te_test,mo_test


def select_random_data(total_points,Nf,X_test,T_test,Te_test,Mo_test):
    id_f = np.random.choice(total_points, Nf, replace=False)# Randomly chosen points for Interior
    print(id_f)
    X_data_Nu = X_test[id_f]
    T_data_Nu = T_test[id_f]
    Te_data_Nu = Te_test[id_f]
    Mo_data_Nu = Mo_test[id_f]
    print("We have",total_points,"points. We will select",X_data_Nu.shape[0],"data points and to train our model.")
    return X_data_Nu,T_data_Nu,Te_data_Nu,Mo_data_Nu