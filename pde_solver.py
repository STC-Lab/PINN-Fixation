#Solve the biharmonic equation as a coupled pair of diffusion equations.

from fipy import Grid1D, CellVariable, TransientTerm, DiffusionTerm, Viewer
from builtins import range

m = Grid1D(nx=100, Lx=1.)
v = CellVariable(mesh=m, hasOld=True, value=[[0.5], [0.5]], elementshape=(2,))
v.constrain([[0], [1]], m.facesLeft)
v.constrain([[1], [0]], m.facesRight)
eqn = TransientTerm([[1, 0],
                     [0, 1]]) == DiffusionTerm([[[0.01, -1],
                                                 [1, 0.01]]])
vi = Viewer((v[0], v[1]))

for t in range(1):
    v.updateOld()
    eqn.solve(var=v, dt=1.e-3)
    vi.plot()