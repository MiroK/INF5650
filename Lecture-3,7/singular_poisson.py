"""
Solve 
-u`` = f in (0, 1)
u`(0) = u`(1) = 0

a) Show that the matrix is as expected
b) Show that the problem is singular

"""

from dolfin import *

def singular_poisson(k):
  """Test solvers and rhs for singular Poisson problem."""

  mesh = UnitIntervalMesh(1000)
  V = FunctionSpace(mesh, "CG", 1)

  u = TrialFunction(V)
  v = TestFunction(V)

  f = Expression("sin(k*pi*x[0])", k=k)

  a = inner(grad(u), grad(v))*dx
  L = f*v*dx

  u = Function(V)
  A, b = assemble_system(a, L)

  #print A.array()
  print "Necessary condition holds for %d?" %k,
  print "Yes\n" if near(assemble(f*dx, mesh=mesh), 0) else "No\n" 

  try:
    solve(A, u.vector(), b, "cg")
    plot(u, interactive=True, title="Iterative with k=%d" % k)
    print "Iterative found solution with mean", assemble(u*dx), u.vector().sum()
  except RuntimeError:
    print "Iterative diverged"

  print
  u.vector().zero()

  try:
    solve(A, u.vector(), b, "lu")
    plot(u, interactive=True, title="Direct with k=%d" % k)
    print "Direct found solution with mean", assemble(u*dx), u.vector().sum()
  except RuntimeError:
    print "Direct diverged"


singular_poisson(k=2)         # Direct does not throw but the solution is wrong
