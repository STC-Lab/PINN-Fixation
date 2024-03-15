import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import loss
import gradient

def move_batch_to_device(batch, device="cpu"):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

class Callback:
    """
    Callback base class which allows for bare functionality of Trainer
    """
    def __init__(self):
        pass

    def begin_train(self, trainer):
        pass

    def begin_epoch(self, trainer, output):
        pass

    def begin_eval(self, trainer, output):
        pass

    def end_batch(self, trainer, output):
        pass

    def end_eval(self, trainer, output):
        pass

    def end_epoch(self, trainer, output):
        pass

    def end_train(self, trainer, output):
        pass

    def begin_test(self, trainer):
        pass

    def end_test(self, trainer, output):
        pass





class Trainer:

    def __init__(self,
                 net1,
                 net2,
                 train_data: torch.utils.data.DataLoader,
                 optimizer1,
                 optimizer2,
                 epochs,
                 epoch_verbose,
                 callback = Callback()):
        
        self.model1 = net1
        self.model2 = net2
        self.optimizer1 = optimizer1
        self.optimizer2 = optimizer2
        self.train_data = train_data
        self.epochs = epochs
        self.current_epoch = 0
        self.epoch_verbose = epoch_verbose
        self.device = 'cpu'
        self.input_keys_net1 = ['x','t']
        self.input_keys_net2 = ['u_hat','du_dt','du_dx','d2u_dx2']
        self.output_keys = ['u_hat']
        self.clip = 100
        self.callback = callback
        


    def train(self):
        
        self.callback.begin_train(self)
        writer = SummaryWriter()
        #Trainable Parameters
        #lamb = torch.nn.Parameter(torch.tensor(1.0))  
        #nu = torch.nn.Parameter(torch.tensor(1.0))           
        #ga = torch.nn.Parameter(torch.tensor(1.0))

        try:
            for i in range(self.current_epoch,self.current_epoch+self.epochs):
                self.model1.train()
                self.model2.train()
                losses = []
                for t_batch in self.train_data:
                    
                    #update the batch
                    t_batch['epoch'] = i
                    t_batch = move_batch_to_device(t_batch, self.device)
                    inputs = [t_batch[k] for k in self.input_keys_net1]
                    #print(inputs)
                    u_hat = self.model1(*inputs)                     #output of the network
                    t_batch['u_hat'] = u_hat
                    t_batch['du_dt'] = gradient.gradient(t_batch['u_hat'],t_batch['t'])
                    t_batch['du_dx'] = gradient.gradient(t_batch['u_hat'],t_batch['x'])
                    t_batch['d2u_dx2'] = gradient.gradient(t_batch['du_dx'],t_batch['x'])
                    #print('t_batch:',t_batch)
                    #loss and update parameters in loss
                    data_loss = loss.network_loss(t_batch)

                    inputs2 = [t_batch[k] for k in self.input_keys_net2]
                    zero = self.model2(*inputs2)
                    print(zero)
                    t_batch['zero'] = zero 
                    pde_loss = loss.pde_loss(t_batch)

                    tol_loss = data_loss+pde_loss
                    self.optimizer1.zero_grad()
                    self.optimizer2.zero_grad() 
                    tol_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model1.parameters(), self.clip)
                    torch.nn.utils.clip_grad_norm_(self.model2.parameters(), self.clip)

                    self.optimizer1.step()
                    self.optimizer2.step()
                    losses.append(tol_loss)


                    #pde_loss = loss.PLoss()
                    #pde_loss.train()
                    #p_loss = pde_loss(t_batch)
                    #p_loss.backward()
                    #self.optimizer.step()


                    self.callback.end_batch(self, t_batch)
                    #print(loss.PLoss.para)

                t_batch['loss'] = torch.mean(torch.stack(losses)) 
                writer.add_scalar('train_loss',t_batch['loss'],i)
                self.callback.begin_epoch(self,t_batch)  





                if i % (self.epoch_verbose) == 0:
                    mean_loss = t_batch['loss']
                    print(f'epoch: {i}  train loss: {mean_loss}')  

        except KeyboardInterrupt:
            print("Interrupted training loop.")

        return self.model1,self.model2

