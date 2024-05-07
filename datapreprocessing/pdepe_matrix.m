x = linspace(0,1,128);
t = linspace(0,1,64);

%sol = pdepe(m,@pdex1pde,@pdex1ic,@pdex1bc,x,t);

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


surf(x,t,u2,'FaceColor','interp','EdgeColor','none');
title('u_2(x,t)');
xlabel('Distance x');
ylabel('Time t');

function [c,f,s] = pdefun(x,t,u,dudx) % Equation to solve
c = [1;1];
f = [3.5;5.5].* dudx;
%s = [3*u(1)+6*u(2);7*u(1)+9*u(2)];     %s1

%s = [-F; F];  %[-1, 1;1,-1]
%s = [6.7;4.7].*dudx+[8.5*u(1);2.5*u(2)];       %s2
%s = [4.7;8.7].*dudx+[8.5*u(1);6.5*u(2)];
s = [8.5*x*u(1);0*u(2)];
end
% ---------------------------------------------
function u0 = pdeic(x) % Initial Conditions
u0 = [sin(0.5*pi*x); sin(0.5*pi*x)];
%u0 = [cos(pi*x); cos(pi*x)];
end
% ---------------------------------------------
function [pl,ql,pr,qr] = pdebc(xl,ul,xr,ur,t)
pl = [0; ul(2)];
ql = [1; 0];
pr = [ur(1)-1; 0];
qr = [0; 1];
end
% ---------------------------------------------