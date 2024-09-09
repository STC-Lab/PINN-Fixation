clear all;
%x = linspace(0,1,80);
%data = load('paper.mat');
%x = data.data.x;
%x = x';
%file1 = '3matrix_test.json';
%jsonText = fileread(file1);
%data_pre = jsondecode(jsonText);
%t = linspace(0,5,30);
%x = [0,0.1464,0.5,0.8536,1];
%x = linspace(-5,5,30);
%sol = pdepe(m,@pdex1pde,@pdex1ic,@pdex1bc,x,t);
%x = data_pre.x;
%t = data_pre.t;
m=0;         
sol = pdepe(m,@pdefun,@pdeic,@pdebc,x,t);
u1 = sol(:,:,1);
u2 = sol(:,:,2);
usol1 = u1';
usol2=u2';


surf(x,t,u1,'FaceColor','interp','EdgeColor','none');
title('u_1(x,t)');
xlabel('Distance x');
ylabel('Time t');
colorbar;

surf(x,t,u2,'FaceColor','interp','EdgeColor','none');
title('u_2(x,t)');
xlabel('Distance x');
ylabel('Time t');
colorbar;

function [c,f,s] = pdefun(x,t,u,dudx) % Equation to solve
c = [1;1];
%f = [1.2;0.8].* dudx;
%s = [3*u(1)+6*u(2);7*u(1)+9*u(2)];     %s1
%f = [(x+2)^3*dudx(1)-x^2*dudx(2);x*dudx(1)+(x+2)^2*dudx(2)];
%f = [x*dudx(1)+x^2*dudx(2);3*x*dudx(1)+4*x*dudx(2)];
f = [3*dudx(1)-1.1*dudx(2);2.7*dudx(1)+1.5*dudx(2)];
%f = [2500*dudx(1)+250*dudx(2)+2dudx(3);30*dudx(1)+5*dudx(2)+dudx(3)];
%s = [-F; F];  %[-1, 1;1,-1]
%s = [6.7;4.7].*dudx+[8.5*u(1);2.5*u(2)];       %s2
%s = [4.7;8.7].*dudx+[8.5*u(1);6.5*u(2)];
s = [2*u(1)+1.8*u(2);1.2*u(1)-2.5*u(2)];
%s = [1.2*dudx(1)+2.5*dudx(2);3.5*dudx(1)+1.5*dudx(2)] + [2*u(1)+1.8*u(2);1.2*u(1)+2.5*u(2)];
%s=[10*sin(10*t);10*sin(10*t)];
%s = [500*u(1)+20*u(2)+1*u(3);100*u(1)+1.7*u(2)+1*u(3)];
end
% ---------------------------------------------
function u0 = pdeic(x) % Initial Conditions
u0 = [sin(0.5*pi*x); sin(0.5*pi*x)];
%u0 = [sin(0.5*pi*x); sin(0.5*pi*x);sin(0.5*pi*x)];
%u0 = [1;1];
end
% ---------------------------------------------
function [pl,ql,pr,qr] = pdebc(xl,ul,xr,ur,t)
%pl = [0; 0];
%ql = [1; 1];
%pr = [ur(1)-1; ur(2)-1];
%qr = [0; 0];
pl = [0; ul(2)];
ql = [1; 0];
pr = [ur(1)-1; 0];
qr = [0; 1];

end
% ---------------------------------------------