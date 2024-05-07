data = load('fixation_data.mat');
output = data.output.timedep;
t = output.dtime;
x = linspace(0,1,8);
u = output.pde;
for i = 1:200
    
    u1(i,1) = u(:,:,i)[]
end
