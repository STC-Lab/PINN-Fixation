u1 = csvread('u1.csv')';
u2 = csvread('u2.csv')';

x = linspace(0,1,128);
t = linspace(0,1,64);


surf(x,t,u1,'FaceColor','interp','EdgeColor','none');
title('u_1(x,t)');
xlabel('Distance x');
ylabel('Time t');


surf(x,t,u2,'FaceColor','interp','EdgeColor','none');
title('u_2(x,t)');
xlabel('Distance x');
ylabel('Time t');