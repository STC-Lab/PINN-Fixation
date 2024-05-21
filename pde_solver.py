#Solve the biharmonic equation as a coupled pair of diffusion equations.

from fipy import Grid1D, CellVariable, TransientTerm, DiffusionTerm, Viewer
from builtins import range

m = Grid1D(nx=100, Lx=1.)
# v = CellVariable(mesh=m, hasOld=True, value=[[0.5], [0.5]], elementshape=(2,))
# v.constrain([[0], [1]], m.facesLeft)
# v.constrain([[1], [0]], m.facesRight)
# eqn = TransientTerm([[1, 0],
#                      [0, 1]]) == DiffusionTerm([[[0.01, -1],
#                                                  [1, 0.01]]])
# vi = Viewer((v[0], v[1]))

# for t in range(1):
#     v.updateOld()
#     eqn.solve(var=v, dt=1.e-3)
#     vi.plot()
v0 = CellVariable(mesh=m, hasOld=True, value=0.5)
v1 = CellVariable(mesh=m, hasOld=True, value=0.5)
v0.constrain(0, m.facesLeft)
v0.constrain(1, m.facesRight)
v1.constrain(1, m.facesLeft)
v1.constrain(0, m.facesRight)
vi = Viewer((v0, v1))


v0.value = 0.5
v1.value = 0.5
eqn0 = TransientTerm(var=v0) == DiffusionTerm(0.5, var=v0) - DiffusionTerm(1.5, var=v1)
eqn1 = TransientTerm(var=v1) == DiffusionTerm(2.2, var=v0) + DiffusionTerm(1.8, var=v1)
eqn = eqn0 & eqn1
from builtins import range
for t in range(1):
    v0.updateOld()
    v1.updateOld()
    eqn.solve(dt=1.e-3)
    vi.plot()


# from pde import PDE, FieldCollection, PlotTracker, ScalarField, UnitGrid

# # define the PDE
# a, b = 1, 3
# d0, d1 = 1, 0.1
# eq = PDE(
#     {
#         "u": f"{d0} * laplace(u) + {a} - ({b} + 1) * u + u**2 * v",
#         "v": f"{d1} * laplace(v) + {b} * u - u**2 * v",
#     }
# )

# # initialize state
# grid = UnitGrid([64, 64])
# u = ScalarField(grid, a, label="Field $u$")
# v = b / a + 0.1 * ScalarField.random_normal(grid, label="Field $v$")
# state = FieldCollection([u, v])

# # simulate the pde
# tracker = PlotTracker(interrupts=1, plot_args={"vmin": 0, "vmax": 5})
# sol = eq.solve(state, t_range=20, dt=1e-3, tracker=tracker)
