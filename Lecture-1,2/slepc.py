from dolfin import *

# Solve the eigenvalue problem
# -u`` = lmbda u      in (0, 1)
# u(0) = u(1) = 1
# but this time use petsc instead of pytave/oct2py

mesh = UnitIntervalMesh(500)
V = FunctionSpace(mesh, "Lagrange", 1)

u = TrialFunction(V)
v = TestFunction(V)
a = dot(grad(u), grad(v))*dx
m = u*v*dx
L = Constant(0)*v*dx

bc = DirichletBC(V, Constant(0), DomainBoundary())

A = PETScMatrix()
M = PETScMatrix()
assemble_system(a, L, bc, A_tensor=A)
assemble_system(m, L, bc, A_tensor=M)

# Create eigensolver
eigensolver = SLEPcEigenSolver(A, M)

# Compute all eigenvalues of A x = \lambda x
print "Computing eigenvalues. This can take a minute."
eigensolver.solve()

# plot all the real eigenvalues
from numpy import zeros
import pylab as py

N = V.dim()
r_eigs = zeros(N)
for i in range(N):
  r, c, rx, cx = eigensolver.get_eigenpair(i)
  r_eigs[i] = r

r_eigs.sort()
py.loglog(r_eigs, label="real eigenvalues")
py.legend(loc='best')
py.show()
