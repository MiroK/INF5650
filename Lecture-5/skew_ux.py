"""
Show that Lu = u_x leads to show symmetric matrix with CG1.
"""

from dolfin import *

mesh = UnitIntervalMesh(10)

V = FunctionSpace(mesh, 'CG', 1)

u = TrialFunction(V)
v = TestFunction(V)

bc = DirichletBC(V, Constant(0), DomainBoundary())
a = u.dx(0)*v*dx
L = Constant(0)*v*dx
A_, b = assemble_system(a, L, bc)

A = A_.array()[1:-1, 1:-1]   # exlude the bc dofs

print "Is skew symmteric", near((A.T + A).sum(), 0)
