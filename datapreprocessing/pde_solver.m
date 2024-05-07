x = linspace(0,2,128);
t = linspace(0,1,64);

%sol = pdepe(m,@pdex1pde,@pdex1ic,@pdex1bc,x,t);

m=0;         
sol = pdepe(m,@pdex1pde,@pdex1ic,@pdex1bc,x,t);
u = sol(:,:,1);
usol = u';
surf(x,t,u)
xlabel('x')
ylabel('t')
zlabel('u(x,t)')
view([150 25])

function [c,f,s] = pdex1pde(x,t,u,dudx)
c = 1;
f = 3.5*dudx;
%s = 6.7*dudx+8.5*u;
s = 6.7*dudx+8.5*u;
end

function u0 = pdex1ic(x)
u0 = sin(pi*x);
end

function [pl,ql,pr,qr] = pdex1bc(xl,ul,xr,ur,t)
pl = ul; %ignored by solver since m=1
ql = 0; %ignored by solver since m=1
pr = ur;
qr = 0; 
end