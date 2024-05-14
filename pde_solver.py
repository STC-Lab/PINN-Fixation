# from fenics import *
# import numpy as np
# import matplotlib.pyplot as plt

# # 创建网格和函数空间
# nx = 50
# mesh = IntervalMesh(nx, 0, 1)
# V = FunctionSpace(mesh, 'P', 1)

# # 定义边界条件
# u_D = Constant(0)

# def boundary(x, on_boundary):
#     return on_boundary

# bc = DirichletBC(V, u_D, boundary)

# # 定义初始条件
# u_init = Expression(('sin(pi*x[0])', 'cos(pi*x[0])'), degree=2)
# u_n = interpolate(u_init, V)
# v_n = interpolate(u_init, V)

# # 定义试探函数和检验函数
# u = TrialFunction(V)
# v = TestFunction(V)

# # 定义时间步长和终止时间
# dt = 0.01
# T = 2.0

# # 定义参数矩阵
# alpha = np.array([[1.0, 0.5], [0.5, 1.0]])
# beta = np.array([[0.1, 0.2], [0.3, 0.4]])

# # 定义方程
# a11 = alpha[0,0] * dot(grad(u), grad(v)) * dx
# a12 = alpha[0,1] * dot(grad(u), grad(v)) * dx
# a21 = alpha[1,0] * dot(grad(u), grad(v)) * dx
# a22 = alpha[1,1] * dot(grad(u), grad(v)) * dx

# m11 = u*v*dx
# m12 = u*v*dx
# m21 = u*v*dx
# m22 = u*v*dx

# b1 = beta[0,0] * u * v * dx + beta[0,1] * u * v * dx
# b2 = beta[1,0] * u * v * dx + beta[1,1] * u * v * dx

# A = a11 + a12 + a21 + a22
# M = m11 + m12 + m21 + m22
# B = b1 + b2

# # 时间循环
# t = 0
# u = Function(V)
# while t < T:
#     t += dt
#     solve(A == M + B, u, bc)
#     u_n.assign(u)

# # 绘制结果
# plot(u)
# plt.xlabel('x')
# plt.ylabel('u')
# plt.title('Solution at t = {}'.format(t))
# plt.show()




import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# 网格和时间参数
nx = 50
x = np.linspace(0, 1, nx)
dx = x[1] - x[0]
dt = 0.01
t_span = (0, 1)
t_eval = np.linspace(0, 1, 100)

# 初始条件
u1_init = np.sin(np.pi * x)
u2_init = np.cos(np.pi * x)
u_init = np.concatenate([u1_init, u2_init])

# 系数矩阵
alpha = np.array([[1.0, 0.5], [0.5, 1.0]])
beta = np.array([[0.1, 0.2], [0.3, 0.4]])

# 定义偏微分方程
def pde_system(t, u):
    u1 = u[:nx]
    u2 = u[nx:]
    d2u1dx2 = np.gradient(np.gradient(u1, dx), dx)
    d2u2dx2 = np.gradient(np.gradient(u2, dx), dx)
    du1dt = alpha[0,0] * d2u1dx2 + alpha[0,1] * d2u2dx2 + beta[0,0] * u1 + beta[0,1] * u2
    du2dt = alpha[1,0] * d2u1dx2 + alpha[1,1] * d2u2dx2 + beta[1,0] * u1 + beta[1,1] * u2
    return np.concatenate([du1dt, du2dt])

# 使用solve_ivp求解
sol = solve_ivp(pde_system, t_span, u_init, method='RK45', t_eval=t_eval)

# 提取解
u1_sol = sol.y[:nx, :]
u2_sol = sol.y[nx:, :]

# 绘制结果
X, T = np.meshgrid(x, sol.t)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 绘制u1
c1 = ax1.contourf(X, T, u1_sol.T, 20, cmap='viridis')
fig.colorbar(c1, ax=ax1)
ax1.set_title('u1(x, t)')
ax1.set_xlabel('x')
ax1.set_ylabel('t')

# 绘制u2
c2 = ax2.contourf(X, T, u2_sol.T, 20, cmap='viridis')
fig.colorbar(c2, ax=ax2)
ax2.set_title('u2(x, t)')
ax2.set_xlabel('x')
ax2.set_ylabel('t')

plt.tight_layout()
plt.show()

