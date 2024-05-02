# May 2nd,2024

1. The matlab function which used to make dataset seems can only solve the PDE with second derivate and the solution u, if we use add the first deriavete to PDE, the dataset is not accurate which leads to inaccuarate results of network. 
2. If we have two diagonal matrces parameters alpha and gamma in the PDE with the size of 2*2, the network can find the parameters when the gamma[2][2] equals zero. But the approximate solution u_hat from the network seems same as the true solution(see the file results). I think probably can also try least square find the parameters.
3. Can't find a good toobox to solve PDE with parameters full matrix for now, so it's hard to make dataset. But it can make dataset with the sapcial items x in the matrix. I can try to solve that next step.