from scipy.io import loadmat
import torch
import torch.nn as nn  
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from sklearn.model_selection import train_test_split
import pyDOE 



#single output pinn dataprocessing
def load_data(file):
    data = loadmat(file) 
    x = data['x']                                   # space:      256 points between -1 and 1 [256x1]
    t = data['t']                                   # time:       100 time points between 0 and 1 [100x1] 
    # u = data['T']
    u = data['usol']  
    return x,t,u   


def totensor(x,t,u):
    X, T = np.meshgrid(x,t)                         # makes 2 arrays X and T such that u(X[i],T[j])=usol[i][j] are a tuple
    X = torch.tensor(X.T).float()
    T = torch.tensor(T.T).float()
    U = torch.tensor(u).float()
    return X,T,U


def reshape_data(x,t,u):
    x_test = x.reshape(-1,1)
    t_test = t.reshape(-1,1)
    u_test = u.reshape(-1,1)
    return x_test,t_test,u_test


def select_data(total_points,Nf,X_test,T_test,U_test):
    id_f = np.random.choice(total_points, Nf, replace=False)# Randomly chosen points for Interior
    print(id_f)
    X_data_Nu = X_test[id_f]
    T_data_Nu = T_test[id_f]
    U_data_Nu = U_test[id_f]
    print("We have",total_points,"points. We will select",X_data_Nu.shape[0],"data points and to train our model.")
    return X_data_Nu,T_data_Nu,U_data_Nu


def split_data(X_data_tensor,T_data_tensor,U_data_tensor,Nf):
    total_points_v = len(X_data_tensor)
    id = list(range(0, total_points_v))
    id_v = np.random.choice(total_points_v, Nf, replace=False)# Randomly chosen points for Interior
    id_t = [x for x in id if x not in id_v]
    X_data_tensor_train = X_data_tensor[id_t]
    T_data_tensor_train = T_data_tensor[id_t]
    U_data_tensor_train = U_data_tensor[id_t]
    X_data_tensor_val = X_data_tensor[id_v]
    T_data_tensor_val = T_data_tensor[id_v]
    U_data_tensor_val = U_data_tensor[id_v]
    return X_data_tensor_train,X_data_tensor_val,T_data_tensor_train,T_data_tensor_val,U_data_tensor_train,U_data_tensor_val








#mo_pinn dataprocessing

def load_matrix(file):
    data = loadmat(file)
    x = data['x']                                   # space:      256 points between -1 and 1 [256x1]
    t = data['t']                                   # time:       100 time points between 0 and 1 [100x1] 
    # u1 = data['u1']
    # u2 = data['u2'] 
    u1 = data['usol1']
    u2 = data['usol2'] 
    # u3 = data['T3']
    # u4 = data['T4'] 
    # u5 = data['T5'] 
    # return x,t,u1,u2,u3,u4,u5
    return x,t,u1,u2


def matrix_totensor(x,t,u1,u2):
    X, T = np.meshgrid(x,t)                         # makes 2 arrays X and T such that u(X[i],T[j])=usol[i][j] are a tuple
    X = torch.tensor(X.T).float()
    T = torch.tensor(T.T).float()
    U1 = torch.tensor(u1).float()
    U2 = torch.tensor(u2).float()
    # U3 = torch.tensor(u3).float()
    # U4 = torch.tensor(u4).float()
    # U5 = torch.tensor(u5).float()
    return X,T,U1,U2


def reshape_matrix(x,t,u1,u2):
    x_test = x.reshape(-1,1)
    t_test = t.reshape(-1,1)
    u1_test = u1.reshape(-1,1)
    u2_test = u2.reshape(-1,1)
    # u3_test = u3.reshape(-1,1)
    # u4_test = u4.reshape(-1,1)
    # u5_test = u5.reshape(-1,1)
    return x_test,t_test,u1_test,u2_test



def full_data_matrix(total_points,Nf,X_test,T_test,U1_test,U2_test):
    id = list(range(0, total_points))
    id_f = np.random.choice(total_points, Nf, replace=False)# Randomly chosen points for Interior
    X_train_Nu = X_test[id_f]
    T_train_Nu = T_test[id_f]
    U1_train_Nu = U1_test[id_f]
    U2_train_Nu = U2_test[id_f]
    # U3_train_Nu = U3_test[id_f]
    # U4_train_Nu = U4_test[id_f]
    # U5_train_Nu = U5_test[id_f]
    print("We have",total_points,"points. We will select",X_train_Nu.shape[0],"points to train our model.")
    return X_train_Nu,T_train_Nu,U1_train_Nu,U2_train_Nu


def split_matrix(X_data_tensor,T_data_tensor,U1_data_tensor,U2_data_tensor,Nf_val):
    np.random.seed(4321)
    total_points_v = len(X_data_tensor)
    id = list(range(0, total_points_v))
    id_v = np.random.choice(total_points_v, Nf_val, replace=False)# Randomly chosen points for Interior
    id_t = [x for x in id if x not in id_v]
    X_data_tensor_train = X_data_tensor[id_t]
    T_data_tensor_train = T_data_tensor[id_t]
    U1_data_tensor_train = U1_data_tensor[id_t]
    U2_data_tensor_train = U2_data_tensor[id_t]
    X_data_tensor_val = X_data_tensor[id_v]
    T_data_tensor_val = T_data_tensor[id_v]
    U1_data_tensor_val = U1_data_tensor[id_v]
    U2_data_tensor_val = U2_data_tensor[id_v]
    return X_data_tensor_train,X_data_tensor_val,T_data_tensor_train,T_data_tensor_val,U1_data_tensor_train,U1_data_tensor_val,U2_data_tensor_train,U2_data_tensor_val







#fixation dataprocessing

def load_paper(file):
    data = loadmat(file) 
    x = data['x']                                   # space:      256 points between -1 and 1 [256x1]
    t = data['t']                                   # time:       100 time points between 0 and 1 [100x1] 
    u1 = data['T1']
    u2 = data['T2'] 
    u3 = data['T3']
    u4 = data['T4'] 
    u5 = data['T5'] 
    return x,t,u1,u2,u3,u4,u5


def paper_totensor(x,t,u1,u2,u3,u4,u5):
    X, T = np.meshgrid(x,t)                         # makes 2 arrays X and T such that u(X[i],T[j])=usol[i][j] are a tuple
    X = torch.tensor(X.T).float()
    T = torch.tensor(T.T).float()
    U1 = torch.tensor(u1).float()
    U2 = torch.tensor(u2).float()
    U3 = torch.tensor(u3).float()
    U4 = torch.tensor(u4).float()
    U5 = torch.tensor(u5).float()
    return X,T,U1,U2,U3,U4,U5


def reshape_paper(x,t,u1,u2,u3,u4,u5):
    x_test = x.reshape(-1,1)
    t_test = t.reshape(-1,1)
    u1_test = u1.reshape(-1,1)
    u2_test = u2.reshape(-1,1)
    u3_test = u3.reshape(-1,1)
    u4_test = u4.reshape(-1,1)
    u5_test = u5.reshape(-1,1)
    return x_test,t_test,u1_test,u2_test,u3_test,u4_test,u5_test



def full_data_paper(total_points,Nf,X_test,T_test,U1_test,U2_test,U3_test,U4_test,U5_test):
    id = list(range(0, total_points))
    id_f = np.random.choice(total_points, Nf, replace=False)# Randomly chosen points for Interior
    X_train_Nu = X_test[id_f]
    T_train_Nu = T_test[id_f]
    U1_train_Nu = U1_test[id_f]
    U2_train_Nu = U2_test[id_f]
    U3_train_Nu = U3_test[id_f]
    U4_train_Nu = U4_test[id_f]
    U5_train_Nu = U5_test[id_f]
    print("We have",total_points,"points. We will select",X_train_Nu.shape[0],"points to train our model.")
    return X_train_Nu,T_train_Nu,U1_train_Nu,U2_train_Nu,U3_train_Nu,U4_train_Nu,U5_train_Nu


def split_paper(X_data_tensor,T_data_tensor,U1_data_tensor,U2_data_tensor,U3_data_tensor,U4_data_tensor,U5_data_tensor,Nf):
    total_points_v = len(X_data_tensor)
    id = list(range(0, total_points_v))
    id_v = np.random.choice(total_points_v, Nf, replace=False)# Randomly chosen points for Interior
    id_t = [x for x in id if x not in id_v]
    X_data_tensor_train = X_data_tensor[id_t]
    T_data_tensor_train = T_data_tensor[id_t]
    U1_data_tensor_train = U1_data_tensor[id_t]
    U2_data_tensor_train = U2_data_tensor[id_t]
    U3_data_tensor_train = U3_data_tensor[id_t]
    U4_data_tensor_train = U4_data_tensor[id_t]
    U5_data_tensor_train = U5_data_tensor[id_t]
    X_data_tensor_val = X_data_tensor[id_v]
    T_data_tensor_val = T_data_tensor[id_v]
    U1_data_tensor_val = U1_data_tensor[id_v]
    U2_data_tensor_val = U2_data_tensor[id_v]
    U3_data_tensor_val = U3_data_tensor[id_v]
    U4_data_tensor_val = U4_data_tensor[id_v]
    U5_data_tensor_val = U5_data_tensor[id_v]
    return X_data_tensor_train,X_data_tensor_val,T_data_tensor_train,T_data_tensor_val,U1_data_tensor_train,U1_data_tensor_val,U2_data_tensor_train,U2_data_tensor_val,U3_data_tensor_train,U3_data_tensor_val,U4_data_tensor_train,U4_data_tensor_val,U5_data_tensor_train,U5_data_tensor_val




def load_matrix_test(file):
    data = loadmat(file) 
    x = data['x']                                   # space:      256 points between -1 and 1 [256x1]
    t = data['t']                                   # time:       100 time points between 0 and 1 [100x1] 
    return x,t


def matrix_totensor_test(x,t):
    X, T = np.meshgrid(x,t)                         # makes 2 arrays X and T such that u(X[i],T[j])=usol[i][j] are a tuple
    X = torch.tensor(X.T).float()
    T = torch.tensor(T.T).float()
    return X,T


def reshape_matrix_test(x,t):
    x_test = x.reshape(-1,1)
    t_test = t.reshape(-1,1)
    return x_test,t_test