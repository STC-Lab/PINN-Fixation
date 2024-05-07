clc;
clear all;

data = load('fixation_1.mat');
timestep = data.data.timestep;
x1 = data.data.x1;
x2 = data.data.x2;
T1 = data.data.T1;
m1 = data.data.m1;
T2 = data.data.T2;
m2 = data.data.m2;

T1_flipped = T1(end:-1:1,:);
T2_flipped = T2(end:-1:1,:);
m1_flipped = m1(end:-1:1,:);
m2_flipped = m2(end:-1:1,:);

fixation.x = [x1,x2];
fixation.T = [T1_flipped;T2_flipped];
fixation.M = [m1_flipped;m2_flipped];
fixation.t = timestep;
fixation.Dc = data.data.diffmat;


surf(x1,timestep,T1,'FaceColor','interp','EdgeColor','none');
title('T_1(x_1,t)');
xlabel('Distance x_1');
ylabel('Time t');

surf(x1,timestep,m1,'FaceColor','interp','EdgeColor','none');
title('M_1(x_1,t)');
xlabel('Distance x_1');
ylabel('Time t');

surf(x2,timestep,T2,'FaceColor','interp','EdgeColor','none');
title('T_2(x_2,t)');
xlabel('Distance x_2');
ylabel('Time t');

surf(x2,timestep,T2,'FaceColor','interp','EdgeColor','none');
title('M_2(x_2,t)');
xlabel('Distance x_2');
ylabel('Time t');

