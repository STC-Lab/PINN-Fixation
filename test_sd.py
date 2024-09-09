import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import json

class FNet(nn.Module):
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
        #x = torch.cat(input, dim=-1)
        x = input
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x

    
class GNet(nn.Module):
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
        #x = torch.cat(input, dim=-1)
        x = input
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x
    

x_test = torch.tensor(np.linspace(0, 1, 100).reshape(-1, 1), dtype=torch.float32)

# model = torch.load('E:\yhy_files\graduation\code\Fixation\sd_model/FX_tanhx_newic_MAE_np5000_nd5000_0.001.pkl')
model_fx = torch.load('E:\yhy_files\graduation\code\PINN-Fixation\model/FX_fixation_1to1.pkl')
model_gx = torch.load('E:\yhy_files\graduation\code\PINN-Fixation\model/GX_fixation_1to1.pkl')
# fx = model_fx(x_test)
# gx = model_gx(x_test)

# data = {
#     'fx': fx.detach().numpy().tolist(),
#     # 'gx': gx.detach().numpy().tolist(),
#     'x':x_test.tolist(),
# }


# with open('tanh_best.json',"w") as f:
#     json.dump(data,f,indent=4)


fx = model_fx(x_test)
fx_true = x_test**2
plt.figure(figsize=(8, 6))
plt.plot(x_test, fx_true, label='f(x)')
plt.plot(x_test, fx.detach().numpy(), label='prediction')
plt.title("Comparision with the results")
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.grid(True)
plt.legend()
plt.show()



gx = model_gx(x_test)
gx_true = 2*x_test
plt.figure(figsize=(8, 6))
plt.plot(x_test, gx_true, label='g(x)')
plt.plot(x_test, gx.detach().numpy(), label='prediction')
plt.title("Comparision with the results")
plt.xlabel('$x$')
plt.ylabel('$g(x)$')
plt.grid(True)
plt.legend()
plt.show()

