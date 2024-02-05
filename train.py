import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from loss import PLoss

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
                 net,
                 train_data: torch.utils.data.DataLoader,
                 optimizer,
                 epochs,
                 epoch_verbose,
                 callback = Callback()):
        
        self.model = net
        self.optimizer = optimizer
        self.train_data = train_data
        self.epochs = epochs
        self.current_epoch = 0
        self.epoch_verbose = epoch_verbose
        self.device = 'cpu'
        self.input_keys = ['x','t']
        self.output_keys = ['u_hat']
        self.clip = 100
        self.callback = callback
        self.loss = PLoss()


    def train(self):
        
        self.callback.begin_train(self)
        writer = SummaryWriter()
        #Trainable Parameters
        #lamb = torch.nn.Parameter(torch.tensor(1.0))  
        #nu = torch.nn.Parameter(torch.tensor(1.0))           
        #ga = torch.nn.Parameter(torch.tensor(1.0))

        try:
            for i in range(self.current_epoch,self.current_epoch+self.epochs):
                self.model.train()
                self.loss.train()
                losses = []
                for t_batch in self.train_data:
                    
                    #update the batch
                    t_batch['epoch'] = i
                    t_batch = move_batch_to_device(t_batch, self.device)
                    inputs = [t_batch[k] for k in self.input_keys]
                    u_hat = self.model(*inputs)                     #output of the network
                    t_batch['u_hat'] = u_hat
                    #print('t_batch:',t_batch)


                    #loss and update parameters in loss

                    #net_loss = loss.network_loss(t_batch)
                    
                    combine_loss = self.loss(t_batch)
                    self.optimizer.zero_grad() 
                    combine_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                    self.optimizer.step()
                    losses.append(combine_loss)


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

        except KeyboardInterrupt:
            print("Interrupted training loop.")

