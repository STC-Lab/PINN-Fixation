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
    x = data['x']                                   # space:      256 points between -1 and 1 [256x1]
    t = data['t']                                   # time:       100 time points between 0 and 1 [100x1] 
    u = data['usol'] 
    #x = data['xx']
    #t = data['tt']
    #u = data['u']
    return x,t,u                            # velocitu:   PDE solution [256x100] 

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


def select_random_data(total_points,Nf,X_test,T_test,U_test):
    id_f = np.random.choice(total_points, Nf, replace=False)# Randomly chosen points for Interior
    print(id_f)
    X_data_Nu = X_test[id_f]
    T_data_Nu = T_test[id_f]
    U_data_Nu = U_test[id_f]
    # X_physics_Nu =  [item for item in X_test if item not in X_data_Nu]
    # T_physics_Nu =  [item for item in T_test if item not in T_data_Nu]
    # U_physics_Nu =  [item for item in U_test if item not in U_data_Nu]
    print("We have",total_points,"points. We will select",X_data_Nu.shape[0],"data points and to train our model.")
    return X_data_Nu,T_data_Nu,U_data_Nu



def select_lhs_data(total_points,Nf,X_test,T_test,U_test):
    id_f = pyDOE.lhs(2)# Randomly chosen points for Interior
    print(id_f)
    X_data_Nu = X_test[id_f]
    T_data_Nu = T_test[id_f]
    U_data_Nu = U_test[id_f]
    return X_data_Nu,T_data_Nu,U_data_Nu



def split_data(X_data_tensor,T_data_tensor,U_data_tensor,X_physics_tensor,T_physics_tensor,U_physics_tensor, Nf):
    total_points_v = len(X_data_tensor)
    id = list(range(0, total_points_v))
    id_v = np.random.choice(total_points_v, Nf, replace=False)# Randomly chosen points for Interior
    id_t = [x for x in id if x not in id_v]
    X_data_tensor_train = X_data_tensor[id_t]
    T_data_tensor_train = T_data_tensor[id_t]
    U_data_tensor_train = U_data_tensor[id_t]
    X_physics_tensor_train = X_physics_tensor[id_t]
    T_physics_tensor_train = T_physics_tensor[id_t]
    U_physics_tensor_train = U_physics_tensor[id_t]
    X_data_tensor_val = X_data_tensor[id_v]
    T_data_tensor_val = T_data_tensor[id_v]
    U_data_tensor_val = U_data_tensor[id_v]
    X_physics_tensor_val = X_physics_tensor[id_v]
    T_physics_tensor_val = T_physics_tensor[id_v]
    U_physics_tensor_val = U_physics_tensor[id_v]
    return X_data_tensor_train,X_data_tensor_val,T_data_tensor_train,T_data_tensor_val,U_data_tensor_train,U_data_tensor_val,X_physics_tensor_train,X_physics_tensor_val,T_physics_tensor_train,T_physics_tensor_val,U_physics_tensor_train,U_physics_tensor_val

def full_data(total_points,Nf,X_test,T_test,U_test):
    id = list(range(0, total_points))
    id_f = np.random.choice(total_points, Nf, replace=False)# Randomly chosen points for Interior
    id_p = [x for x in id if x not in id_f]
    X_train_Nu = X_test[id_f]
    T_train_Nu = T_test[id_f]
    U_train_Nu = U_test[id_f]
    X_physics_data = X_test[id_p]
    T_physics_data = T_test[id_p]
    U_physics_data = U_test[id_p]
    print("We have",total_points,"points. We will select",X_train_Nu.shape[0],"points to train our model.")
    return X_train_Nu,T_train_Nu,U_train_Nu,X_physics_data,T_physics_data,U_physics_data

def data_physics(dataset):
    data_point,physics_point = train_test_split(dataset,test_size=0.5)
    return data_point,physics_point

class DictDataset(Dataset):
    """
    Basic dataset compatible with neuromancer Trainer
    """

    def __init__(self, datadict, name='train'):
        """

        :rtype: object
        :param datadict: (dict {str: Tensor})
        :param name: (str) Name of dataset
        """
        super().__init__()
        self.datadict = datadict
        #print(self.datadict.items())
        lens = [v.shape[0] for v in datadict.values()]          #[len(x) len(t) len(u)]
        assert len(set(lens)) == 1, 'Mismatched number of samples in dataset tensors'
        self.length = lens[0]
        self.name = name

    def __getitem__(self, i):
        #print('DictDataset.__getitem__')
        """Fetch a single item from the dataset."""
        return {k: v[i] for k, v in self.datadict.items()}

    def __len__(self):
        #print('DictDataset.__len__')
        return self.length

    def collate_fn(self, batch):
        """Wraps the default PyTorch batch collation function and adds a name field.

        :param batch: (dict str: torch.Tensor) dataset sample.
        """
        #print('DictDataset.collate_fn')
        batch = default_collate(batch)
        #print(batch)
        batch['name'] = self.name
        return batch
    
