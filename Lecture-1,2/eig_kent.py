from dolfin import *
import oct2py as op
from numpy import matrix, diagflat, sqrt
from numpy import linalg
octave = op.Oct2Py()

mesh = UnitIntervalMesh(1000)
V = FunctionSpace(mesh, "Lagrange", 2)
u = TrialFunction(V)
v = TestFunction(V)
bc = DirichletBC(V, Constant(0), DomainBoundary())
A, _ = assemble_system(inner(grad(u), grad(v))*dx, Constant(0)*v*dx, bc)
M, _ = assemble_system(u*v*dx, Constant(0)*v*dx, bc)

# e, l = eigenvectors, eigenvalues
e, l = octave.eig(A.array(), M.array())
e = matrix(e)
l = matrix(l)

k=1
u_ex = Expression("sin(k*pi*x[0])", k=k)
u = interpolate(u_ex, V)
x = matrix(u.vector().array())
ex = e.T*x.T
H1_norm0 = sqrt(assemble(inner(grad(u), grad(u))*dx))
print "H1 norm assemble of sin(%d pi x) %e" % (k, H1_norm0)
H1_norm = sqrt(ex.T*l*ex) / len(ex)
diffH1 = abs(H1_norm - H1_norm0)
print "H1 norm eigen of sin(%d pi x) %e, diff %e " % (k, H1_norm, diffH1)
L2_norm0 =sqrt(assemble(u**2*dx))
print "L2 norm assemble of sin(%d pi x) %e " % (k, L2_norm0)
L2_norm = sqrt(ex.T*l**0*ex) / len(ex)
diffL2 = abs(L2_norm - L2_norm0)
print "L2 norm eigen of sin(%d pi x) %e, diff %e " % (k, L2_norm, diffL2)
