"""
Consider 
-u`` + u  = f  in \Omega
-u`.n = g on \partial\Omega

Investigate positive definitness of resulting matrix.
"""

from dolfin import *
import oct2py as op

mesh = UnitSquareMesh(10, 10)
V = FunctionSpace(mesh, "CG", 2)

u = TrialFunction(V)
v = TestFunction(V)

a = inner(grad(u), grad(v))*dx + u*v*dx

A = assemble(a)

# We've seen that this matrix should be positive definite
# and therefore there should be not 0 eigenvalues in the spectra

octave = op.Oct2Py()

_, lambdas = octave.eig(A.array())
lambdas = lambdas.diagonal()
lambdas.sort()

print "Smallest eigenvalue of A", lambdas[0]
