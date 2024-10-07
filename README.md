# Inplementation

## Environment
```
python == 3.10.4
```

## Strcture

The file 3parafixation.py is used to eatimation the constant number parameters.

The file fixation_mo.py is used to eatimation the constant matrix parameters.

The file fixation_sd.py is used to eatimation the parameters with spatial dependency.

The file fixation_sdmo.py is used to eatimation the matrix parameters with spatial dependency.

The file dataprecessing.py is used to preprocess the data, resize the data, transform the data to tensor and make the validation set.

The file draw.py is used to save the eatimation results using the trained model.

The file test_sd.py is used to visualize the function parameters estimation results.

## File 

In dataset, there has some example datasets.

In model, there has some trained model to test.

In MATLAB, there is the code of pde solver for single number or matrix parameters.


## Implementation
Use 3parafixation.py as an example.
The dataset can be changed here.And Make the training and validation dataset.
```
file = './dataset111/3para_x45_t100.mat'        #The dataset includes x,t and u
x,t,u = dataprocessing.load_data(file)
X,T,U = dataprocessing.totensor(x,t,u)
X_test,T_test,U_test = dataprocessing.reshape_data(X,T,U)
total_points=len(x[0])*len(t[0])
print('The dataset has',total_points,'points')
```
Choose the amout of the data points
```
Nf = 4500 # Nf: Number of collocation points
```
Choose the amount of the physics points, the amount is square of num_samples.
```
num_samples = 63
```
Define the network,(input,output,neurons in each layer,layers)
```
pinn = FCN(2,1,32,3)
```
Define the trainable parameters,the parameters can be defined as matrix.
```
alpha = torch.nn.Parameter(torch.ones(1, requires_grad=True))
beta = torch.nn.Parameter(torch.ones(1, requires_grad=True))
gamma = torch.nn.Parameter(torch.ones(1, requires_grad=True))
```
Add the trainable parameters in the optimizer
```
optimiser1 = torch.optim.Adam(list(pinn.parameters())+[alpha]+[beta]+[gamma],lr=0.001)
```
Write the physics loss function, depends on the model which generate the dataset
```
        physics_input = [X_physics_tensor,T_physics_tensor]
        physic_output = pinn(physics_input)
        dudt = torch.autograd.grad(physic_output, T_physics_tensor, torch.ones_like(physic_output), create_graph=True)[0]
        dudx = torch.autograd.grad(physic_output, X_physics_tensor, torch.ones_like(physic_output), create_graph=True)[0]
        d2udx2 = torch.autograd.grad(dudx, X_physics_tensor, torch.ones_like(dudx), create_graph=True)[0]
        loss1 = torch.mean((alpha*d2udx2+beta*dudx+gamma*physic_output-dudt)**2)
        physicsloss.append(loss1.item())
```
If the parameters are matrix, the loss funxtion are defuned like this.Here, alpha is 2*2 trainable parameters 
```
loss1 = torch.mean((alpha[0][0]*d2u1dx2+alpha[0][1]*d2u2dx2+mu[0][0]*physic_output1+mu[0][1]*physic_output2-du1dt)**2+(alpha[1][0]*d2u1dx2+alpha[1][1]*d2u2dx2+mu[1][0]*physic_output1+mu[1][1]*physic_output2-du2dt)**2)
```

Define the physics loss
```
        data_input = [X_data_tensor_train,T_data_tensor_train]
        data_output = pinn(data_input)
        loss2 = torch.mean((U_data_tensor_train - data_output)**2)
        dataloss.append(loss2.item())
```

For different dataset, the loss function are all different, so we need to change it based on each diffent model.

Save all the results
```
data = {'alpha':alps,
        'beta':bets,
        'gamma':gams,
        'lambda':lamb,
        # 'lambda2':lamb2,
         'physicsloss':physicsloss,
         'dataloss':dataloss,
         'totalloss':totalloss,
         'valloss':valloss,
         'time':time_sum,
        'finalloss':minloss,
         'paraerror': error,
         'finalerror': minerror,
         'index': min_index,
}


with open('3para_phylamb0.01_datalamb1000_init1.json',"w") as f:
    json.dump(data,f,indent=4)

torch.save(pinn,"./model/3para_phylamb0.01_datalamb1000_init1.pkl.")
```

'xxxx.json' save the numerical results in each iteration, and 'xxxx.pkl' save the trained model.
## Example

Here are the example model for test each netwrok. The coresponding dataset is in the file dataset. And the visualization results are in the file results. The trained models are in the file model. 

### single number parameters

![](graph/3para.png)

### matrix parameters

![](graph/3matrix.png)

### function parameters example 1

![](graph/1to1_sd_example1.png)

### function parameters example 2

![](graph/1to1_sd_example2.png)