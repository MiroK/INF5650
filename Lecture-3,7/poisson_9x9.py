"""
Solve

-u`` + u = f in (0, 1)
u(0) = u(1) = 0

a) compare A, M matrix with hand computation
b) compare b vector with hand computation
"""

from dolfin import *
from scipy.linalg import toeplitz             # for generating own A, M
from numpy import zeros, any
from numpy import dot as npdot

mesh = UnitIntervalMesh(10)                   # triangulation
V = FunctionSpace(mesh, 'CG', 1)              # V_h

u = TrialFunction(V)
v = TestFunction(V)

f = Expression("sin(x[0])")

a = inner(grad(u), grad(v))*dx                # stiffness matrix form
m = u*v*dx                                    # mass matrix form
L = f*v*dx                                    # load vector form

bc = DirichletBC(V, Constant(0), DomainBoundary())  # homog. boundary condition

# check the matrices
h = mesh.hmin()
A = assemble(a)
M = assemble(m)
bc.apply(A); bc.apply(M)  # the matrices include entries for phi_0, phi_10

# check the load vector first, for later we will cancel h
b = assemble(L)
bc.apply(b)

F = interpolate(f, V).vector().array()
b_hand = npdot(M.array(), F)
b_hand[0] = 0; b_hand[-1] = 0         # set the bcs

b[:] -= b_hand
print "Is b okay", b.norm("l1") < 1E-15, "\n"

# remove the phi_0, phi_10 entries for easier comparison
h = mesh.hmin()
A =  A.array()[1:-1, 1:-1]*h
M = M.array()[1:-1, 1:-1]*6/h

print "A", "\n", A, "\n"
print "M", "\n", M, "\n"

# if visual comparison is not enough
A_col = zeros(9); A_col[0] = 2; A_col[1] = -1;
print "A is okay?", any(abs(A - toeplitz(A_col)) > 1E-15)

M_col = zeros(9); M_col[0] = 4; M_col[1] = 1;
print "M is okay?", any(abs(M - toeplitz(M_col)) > 1E-15)

