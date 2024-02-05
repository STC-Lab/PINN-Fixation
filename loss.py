import torch.nn as nn
import gradient
import torch


#Collocation point loss
def network_loss(t_batch):
    loss_func = nn.MSELoss()
    return loss_func(t_batch['u_hat'],t_batch['u'])

#PDE lOSS
def pde_loss(t_batch):
    loss_func = nn.MSELoss()
    length = len(t_batch['u_hat'])
    du_dt = gradient.gradient(t_batch['u_hat'],t_batch['t'])
    du_dx = gradient.gradient(t_batch['u_hat'],t_batch['x'])
    d2u_d2x = gradient.gradient(du_dx,t_batch['x'])
    lamb = t_batch['lamb']
    nu = t_batch['nu']
    ga = t_batch['ga']
    f_pinn = du_dt - lamb*d2u_d2x - nu*du_dx - ga*t_batch['u_hat']
    zero = torch.zeros(length,1)
    return loss_func(f_pinn,zero)


def combine_loss(t_batch):
    PL = PLoss()
    combine_loss = network_loss(t_batch)+PL(t_batch)
    return combine_loss




class PLoss(nn.Module):
    
    def __init__(self):
        super(PLoss,self).__init__()
        self.lamb = torch.nn.Parameter(torch.tensor(1.0),requires_grad= True)  
        self.beta = torch.nn.Parameter(torch.tensor(1.0),requires_grad=True)           
        self.gamma = torch.nn.Parameter(torch.tensor(1.0),requires_grad=True)


    def forward(self,t_batch):
        print(self.lamb)
        #print(self.beta)
        #print(self.gamma)
        loss_func = nn.MSELoss()
        length = len(t_batch['u_hat'])
        du_dt = gradient.gradient(t_batch['u_hat'],t_batch['t'])
        du_dx = gradient.gradient(t_batch['u_hat'],t_batch['x'])
        d2u_d2x = gradient.gradient(du_dx,t_batch['x'])
        f_pinn = du_dt - self.lamb*d2u_d2x - self.beta*du_dx - self.gamma*t_batch['u_hat']
        zero = torch.zeros(length,1)
        #pde_loss = loss_func(f_pinn,zero)   
        return loss_func(f_pinn,zero)+self.network_loss(t_batch)
    
    def network_loss(self,t_batch):
        loss_func = nn.MSELoss()
        return loss_func(t_batch['u_hat'],t_batch['u'])
    




    