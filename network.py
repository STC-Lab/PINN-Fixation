import torch
import torch.nn as nn  
from torchsummary import summary

class Block(nn.Module):
    """
    Canonical abstract class of the block function approximator
    """
    def __init__(self):
        super().__init__()

    #@abstractmethod
    def block_eval(self, x):
        pass

    def forward(self, *inputs):
        """
        Handling varying number of tensor inputs

        :param inputs: (list(torch.Tensor, shape=[batchsize, insize]) or torch.Tensor, shape=[batchsize, insize])
        :return: (torch.Tensor, shape=[batchsize, outsize])
        """
        if len(inputs) > 1:
            x = torch.cat(inputs, dim=-1)
            #print('xshape:',x.shape)
        else:
            x = inputs[0]
        return self.block_eval(x)
    


class Linear(nn.Module):
        
    def __init__(self,insize,outsize):
         super().__init__()
         self.linear = nn.Linear(insize,outsize)
    
    def effective_W(self):
        return self.linear.weight.T

    def forward(self, x):
        return self.linear(x)
    

class Network(Block):
        
    def __init__(
            self,
            input_size,
            output_size,
            hsizes,
            nonlin,
            linear_map = Linear,
            ):
        super().__init__()
        self.insize = input_size
        self.outsize = output_size
        self.nhidden = len(hsizes)
        sizes = [input_size]+hsizes+[output_size]
        self.nonlin = nn.ModuleList([nonlin() for k in range(self.nhidden)]+[nn.Identity()]) 
        self.linear = nn.ModuleList(
            [
                linear_map(sizes[k],sizes[k + 1]) for k in range(self.nhidden+1)
            ]
        )
            
    def reg_error(self):
        return sum([k.reg_error() for k in self.linear if hasattr(k, "reg_error")])

    def block_eval(self, x):
        """

        :param x: (torch.Tensor, shape=[batchsize, insize])
        :return: (torch.Tensor, shape=[batchsize, outsize])
        """
        for lin, nlin in zip(self.linear, self.nonlin):
            x = nlin(lin(x))
        return x        
            

class ParaNet(nn.Module):

    def __init__(self,
                  input_size,
                  output_size):
        super(ParaNet, self).__init__()
        self.net = nn.Linear(input_size,output_size)

    def forward(self,*inputs):
        #print(inputs)
        if len(inputs) > 1:
            x = torch.cat(inputs, dim=-1)
            #print('xshape:',x.shape)
        else:
            x = inputs[0]
        out = self.net(x)
        return out
    

class ConcatNet(nn.Module):
    
    def __init__(self, net1,net2):
        super(ConcatNet,self).__init__()
        self.net1 = net1
        self.net2 = net2
    
    def forward(self,x):
        out = self.net1(x)
        out = self.net2(out)
        return out
        


            

# class Net(torch.nn.Module):

#     def __init__(self,
#                  n_input,
#                  hidden_layer,
#                  n_output,
#                  activate_func):
#         super(Net,self).__init__()
#         self.n_hidden = len(hidden_layer)
#         self.hidden1 = [torch.nn.Linear(n_input,hidden_layer[0])]
#         self.hiddenn = [torch.nn.Linear(hidden_layer[-1],n_output)]
#         self.hidden = self.network(hidden_layer)
#         self.activate_func = activate_func

#     def network(self,hidden_layer):
#         hidden = []
#         if self.n_hidden <= 1:
#             hidden = hidden
#         else:
#             for layer in range(0,self.n_hidden-1):
#                 hidden.append(torch.nn.Linear(hidden_layer[layer],hidden_layer[layer+1]))
#         return hidden
    
#     def forward(self,input):
#         out = self.hidden1(input)
#         out = self.activate_func(out)
#         for layer in self.hidden:
#             out = layer(out)
#             out = self.activate_func(out)
#         out = self.hiddenn(out)
#         return out


# class Net1(Block):

#     def __init__(self, 
#                  input_size, 
#                  hidden_size, 
#                  output_size):
        
#         super(Net1, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_size, output_size)
#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, input):
#         out = self.fc1(input)
#         out = self.relu(out)
#         out = self.fc2(out)
#         out = self.softmax(out)
#         return out
 




        
        
        
