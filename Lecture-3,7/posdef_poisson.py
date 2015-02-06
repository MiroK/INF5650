"""
Consider 
-u`` = f  in (0, 1) with
a) u(0) = u(1) = 0     and
b) u`(0) = u`(1) = 0

Investigate positive definitness of resulting stiffness matrices.
"""

from dolfin import *
import oct2py as op

mesh = UnitIntervalMesh(100)
V = FunctionSpace(mesh, "CG", 2)

u = TrialFunction(V)
v = TestFunction(V)

a = inner(grad(u), grad(v))*dx
bc = DirichletBC(V, Constant(0.), DomainBoundary())

A_dirichlet = assemble(a)
bc.apply(A_dirichlet)

# We've seen that this matrix should be positive definite for homog. Dirichlet 
# conditions and therefore there should be not 0 eigenvalues in the spectra

octave = op.Oct2Py()

_, lambdas = octave.eig(A_dirichlet.array())
lambdas = lambdas.diagonal()
lambdas.sort()

print "Smallest eigenvalue of A_dirichlet", lambdas[0]

# With homog. Neumann condition the matrix is only pos. semidefinite so
# the smallest eigenvalue should be zero or close to it

A_neumann = assemble(a)

_, lambdas = octave.eig(A_neumann.array())
lambdas = lambdas.diagonal()
lambdas.sort()

print "Smallest eigenvalue of A_neumann", lambdas[0]

