x = linspace(0,2,256);
t = linspace(0,1,100);
alpha = rand(1,3);
beta = rand(1,3);
gamma = rand(1,3);

%sol = pdepe(m,@pdex1pde,@pdex1ic,@pdex1bc,x,t);
global ii
global jj
global kk

m = 0;
cell_k = cell(2,length(gamma));
cell_ij = cell(length(alpha),length(beta));
for i = 1:length(alpha)    
    for j = 1:length(beta)
        for k = 1:length(gamma)
            ii = alpha(i);
            jj = beta(j);
            kk = gamma(k);
            sol = pdepe(m,@pdex1pde,@pdex1ic,@pdex1bc,x,t);
            usol = sol';
            cell_k{1,k} = usol;
            cell_k{2,k} = [ii,jj,kk];
%             sol = pdepe(m,@pdex1pde,@pdex1ic,@pdex1bc,x,t);
        end
        cell_ij{i,j} = cell_k;
        cell_k = cell(2,length(gamma));
    end
end
            

            
function [c,f,s] = pdex1pde(x,t,u,dudx)
c = evalin('base','ii');
f = dudx;
s = evalin('base','jj')*dudx+evalin('base','kk')*u
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
            
%sol = pdepe(m,@pdex1pde,@pdex1ic,@pdex1bc,x,t);
%u = sol(:,:,1);
%usol = u';
%surf(x,t,u)
%xlabel('x')
%ylabel('t')
%zlabel('u(x,t)')
%view([150 25])

%function [c,f,s] = pdex1pde(x,t,u,dudx)
%c = 1.6;
%f = dudx;
%s = 3.5*dudx+5.8*u;
%end

%function u0 = pdex1ic(x)
%u0 = sin(pi*x);
%end

%function [pl,ql,pr,qr] = pdex1bc(xl,ul,xr,ur,t)
%pl = ul; %ignored by solver since m=1
%ql = 0; %ignored by solver since m=1
%pr = ur;
%qr = 0; 
%end