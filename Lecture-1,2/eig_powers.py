from dolfin import *
import oct2py as op
from numpy import matrix
from math import sqrt as msqrt
from numpy import dot as npdot

octave = op.Oct2Py()

n_cells = 1000
p_order = 1
mesh = UnitIntervalMesh(n_cells)
V = FunctionSpace(mesh, "Lagrange", p_order)

u = TrialFunction(V)
v = TestFunction(V)
bc = DirichletBC(V, Constant(0), DomainBoundary())
A, _ = assemble_system(inner(grad(u), grad(v))*dx, Constant(0)*v*dx, bc)
M, _ = assemble_system(u*v*dx, Constant(0)*v*dx, bc)

e, l = octave.eig(A.array(), M.array())
e = matrix(e)
l = matrix(l)

u = interpolate(Expression("sin(k*pi*x[0])", k=1), V)
U = matrix(u.vector().array())

# norm from assemble
aL2 = sqrt(assemble(u**2*dx))
aH10 = sqrt(assemble(inner(grad(u), grad(u))*dx))

M = matrix(M.array()) # cast M so that numpy can work with it
eL2, eH10 = 0, 0
for k in range(V.dim()):
  F = matrix(e[:, k])  # select k-th eigenvector
  lmbda = l[k, k]      # select k-th eigenvalue
  eL2 += npdot(npdot(U, M), F)**2
  eH10 += lmbda*(npdot(npdot(U, M), F))**2

eL2 = msqrt(eL2); eH10 = msqrt(eH10)
diffL2 = abs(eL2 - aL2); diffH10 = abs(eH10 - aH10)
print "L2 assemble %g, eigenvalues %g, diff %g" % (aL2, eL2, diffL2)
print "H10 assemble %g, eigenvalues %g, diff %g" % (aH10, eH10, diffH10)
