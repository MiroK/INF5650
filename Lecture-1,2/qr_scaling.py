"""
Making Kent's code work for P2. REQUIRES OCT2PY.
"""

from dolfin import *
import oct2py as op
from numpy import matrix, diagflat, sqrt, diag, array
from numpy import dot as npdot
from numpy import linalg
from random import randrange

octave = op.Oct2Py()

order = 3        # polynomial order for lagrange

for N in [10, 100, 1000]:
  mesh = UnitIntervalMesh(N)

  V = FunctionSpace(mesh, "Lagrange", order)
  u = TrialFunction(V)
  v = TestFunction(V)

  bc = DirichletBC(V, Constant(0), DomainBoundary())
  A, _ = assemble_system(inner(grad(u), grad(v))*dx, Constant(0)*v*dx, bc)
  M, _ = assemble_system(u*v*dx, Constant(0)*v*dx, bc)

  # e, l = eigenvectors, eigenvalues
  e, l = octave.eig(A.array(), M.array())
  e, R = octave.qr(e)
  
  e = matrix(e)
  l = matrix(l)

  h = mesh.hmin()
  k = 10
  u_ex = Expression("sin(k*pi*x[0])", k=k)

  u = interpolate(u_ex, V)
  x = matrix(u.vector().array())
  ex = e.T*x.T

  h = mesh.hmin()

  print "H1-seminorm from assemble", sqrt(assemble(inner(grad(u), grad(u))*dx))
  print "H1-seminorm from eigenvalues", sqrt(ex.T*l*ex/len(ex))
 
  print "L2 norm from assemble", sqrt(assemble(u**2*dx))
  print "L2 norm from eigenvalues",  sqrt(ex.T*l**0*ex/len(ex))

  lm1 = matrix(diag(diag(1./l)))
  print "H^-1 norm from eigenvalues", sqrt(ex.T*lm1*ex/len(ex))

  print 
