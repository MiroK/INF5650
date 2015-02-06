"""
Collection of functions relevant for lectures 1, 2 from Lectures on FEM.
"""

from dolfin import *
import pylab
from numpy import matrix, diag, array, sqrt, zeros
import numpy.random as numpy_random
from numpy import sin  as numpy_sin
from random import sample
from import_tests import *

backends = []
test_oct2py(backends)
test_pytave(backends)
print backends

if not backends:
  print "No backends found for computing eigenvalues. Exiting."
  exit()

#------------------------------------------------------------------------------

def eigenvalue_problem(A, M):
  """Solve the eigenvalue problem with first available backend."""
  if backends[0] == "oct2py":
    import oct2py as op
    octave = op.Oct2Py()
    e, l = octave.eig(A.array(), M.array())
  elif backends[0] == "pytave":
    import pytave
    e, l = pytave.feval(2, "eig", A.array(), M.array())

  return e, l

#------------------------------------------------------------------------------

def eigenvalue_dependency_on_mesh_and_order(Ns, orders):
  """
For meshes with Ns elements and CG-orders function spaces
find the eigenvalues of A, M eigenvalue problem where A
is the stiffness and M is the mass matrix.
"""
  for N in Ns: #Ns:
    mesh = UnitIntervalMesh(N)
    for order in orders:
      V = FunctionSpace(mesh, "Lagrange", order)
      u = TrialFunction(V)
      v = TestFunction(V)

      bc = DirichletBC(V, Constant(0), DomainBoundary())
      A, _ = assemble_system(inner(grad(u), grad(v))*dx, Constant(0)*v*dx, bc)
      M, _ = assemble_system(u*v*dx, Constant(0)*v*dx, bc)

      e, l = eigenvalue_problem(A, M)

      e = matrix(e)
      l = matrix(l)

      x = diag(l)
      x.sort()

      pylab.loglog(x[:], label="[%d, %d]" % (N, order))
      print "Largest eigenvalue for mesh=%d, order=%d is %g" % (N, order, x[-1])

  pylab.title("eigenvalues [mesh, order]")
  pylab.legend(loc='best')
  pylab.show()

#------------------------------------------------------------------------------

def check(value, reference):
  """If value is not near the reference tell how much they differ."""
  passed = near(value, reference)
  if not passed:
    return passed, abs(value - reference)
  else:
    return passed

#------------------------------------------------------------------------------

def eigenvector_orthonormality_test(N, order):
  """
For eigenvectors of A in the A(stifness), M(mass) eigenvalue problem, see
if eigenvectors are orthonormal in L^2 norm.
"""
  print N, order
  mesh = UnitIntervalMesh(N)
  V = FunctionSpace(mesh, "Lagrange", order)
  u = TrialFunction(V)
  v = TestFunction(V)

  bc = DirichletBC(V, Constant(0), DomainBoundary())
  A, _ = assemble_system(inner(grad(u), grad(v))*dx, Constant(0)*v*dx, bc)
  M, _ = assemble_system(u*v*dx, Constant(0)*v*dx, bc)

  e, l = eigenvalue_problem(A, M)

  e = matrix(e)
  l = matrix(l)

  # select random pairs of eigenvector and see whether they are perpend and
  # normalized

  u = Function(V)
  v = Function(V)
  u_v = u.vector()
  v_v = v.vector()
  size = V.dim()
  for test_nr in range(5):
    [i, j] = sample(range(size), 2)
    u_ = array([e[k, i] for k in range(size)])
    v_ = array([e[k, j] for k in range(size)])

    u_v[:] = u_
    v_v[:] = v_

    print "Checking ortho(gona/norma)lity of eigenvectors", i, j
    print "orthogonality in L^2 inner product", check(assemble(u*v*dx), 0.)
    print "orthonormality of %d-th eigenvector in L^2 norm" % i,\
    check(assemble(u*u*dx), 1)
    print "orthonormality of %d-th eigenvector in L^2 norm" % j,\
    check(assemble(v*v*dx), 1)
    # The eigenvectors of continuous eigenproblem are orthogonal in L^2 inner
    # product. Eigenvector of the descrete/matrix eigenproblem are orthogonal
    # in l^2 inner product.
    print

#------------------------------------------------------------------------------

def compare_solution(numeric, exact):
  if exact is None:
    return numeric, numeric
  else:
    if type(numeric) is not float:
      numeric = float(numeric)
    return numeric, abs(exact-numeric)

#------------------------------------------------------------------------------

def get_norms(expression, N, order, exact_solution=None, with_plot=False,
random=0, sin=0):
  """
Given expression approximated on FEM space build in mesh(N...) with
cG-order elements compute its H1-seminorm, L2 and H-1 norms. Compare
with exact solutions.
"""
  mesh = UnitIntervalMesh(N)
  V = FunctionSpace(mesh, "Lagrange", order)

  u = TrialFunction(V)
  v = TestFunction(V)
  bc = DirichletBC(V, Constant(0), DomainBoundary())
  A, _ = assemble_system(inner(grad(u), grad(v))*dx, Constant(0)*v*dx, bc)
  M, _ = assemble_system(u*v*dx, Constant(0)*v*dx, bc)

  e, l = eigenvalue_problem(A, M)

  e = matrix(e)
  l = matrix(l)

  if random:
    u = Function(V)
    if random == 1:
       u.vector()[:] = numpy_random.random(V.dim())
    elif random == -1:
       u.vector()[:] = numpy_random.random(V.dim()) -\
           numpy_random.random(V.dim())
    u.vector()[0] = 0; # put the functions to H10
    u.vector()[-1] = 0;
  elif sin:
    u = Function(V)
    coordinates = mesh.coordinates()
    x = zeros(V.dim())
    if sin == 1:
        for i, y in enumerate(coordinates):
            x[i] = abs(numpy_sin(0.25*N*y))
    elif sin == -1:
        for i, y in enumerate(coordinates):
            x[i] = numpy_sin(0.25*N*y)
    u.vector()[:] = x
    u.vector()[0] = 0; # put the functions to H10
    u.vector()[-1] = 0;
  else:
    u = interpolate(expression, V)

  if with_plot:
    plot(u, interactive=True)

  x = matrix(u.vector().array())
  ex = e.T*x.T

  if exact_solution is None:
    exact_solution = [None, None]

  # norm from assemble
  aH10 = compare_solution(sqrt(assemble(inner(grad(u), grad(u))*dx)), exact_solution[0])
  aL2 = compare_solution(sqrt(assemble(u**2*dx)), exact_solution[1])

  # norm from eigenvalue
  eH10 = compare_solution(sqrt(ex.T*l*ex)/V.dim(), exact_solution[0])
  eL2 = compare_solution(sqrt(ex.T*l**0*ex)/V.dim(), exact_solution[1])
  lm1 = matrix(diag(diag(1./l)))
  eHM1 = compare_solution(sqrt(ex.T*lm1*ex)/V.dim(), None)


  print "H1-seminorm from assemble", aH10
  print "H1-seminorm from eigenvalues", eH10

  print "L2 norm from assemble", aL2
  print "L2 norm from eigenvalues", eL2

  print "H^-1 norm from eigenvalues", eHM1
  print

  # return measure of mesh size and the norms
  return array([mesh.hmin(), aH10[1], aL2[1], eH10[1], eL2[1], eHM1[1]])

#------------------------------------------------------------------------------

class Hat(Expression):
  """
f(x) = x/h - (0.5 - h)/h for x \in [0.5-h, 0.5]
-x/h + (0.5 + h)/h for x \in [0.5, 0.5+h]
"""
  def __init__(self, h):
    Expression.__init__(self)
    self.h = h

  def eval(self, value, x):
    h = self.h
    if (0.5 - h - DOLFIN_EPS) <= x[0] <= (0.5 + DOLFIN_EPS):
      value[0] = x/h - (0.5 - h)/h
    elif (0.5 - DOLFIN_EPS) <= x[0] <= (0.5 + h + DOLFIN_EPS):
      value[0] = -x/h + (0.5 + h)/h
    else:
      value[0] = 0

#------------------------------------------------------------------------------

# code for compiled expression ... f(x) = k*x
kx_code="""
class KX : public Expression
{
public:
double k;

KX() : Expression(){ }

void eval(Array<double>& values, const Array<double>& x) const
{
  values[0] = k*x[0];
}
};"""

#------------------------------------------------------------------------------

# code for compiled expression ... f(x) = x/(1-h) for 0 < x < 1-h,
# (1-x)/h for x < 1-h < 1
kx0_code="""
class KX0 : public Expression
{
public:
double h;

KX0() : Expression(){ }

void eval(Array<double>& values, const Array<double>& x) const
{
if(x[0] < 1 - h - DOLFIN_EPS)
{
  values[0] = x[0]/(1 - h);
}
else
{
  values[0] = (1 - x[0])/h;
}
}
};"""

#------------------------------------------------------------------------------

# compiled expression ... symmetric around 0.5, k steepens the function

kink_code="""
class Kink : public Expression
{
public:
double k;

Kink() : Expression(){ }

void eval(Array<double>& values, const Array<double>& x) const
{
if(x[0] <= 0.5)
{
  values[0] = pow(2*x[0], k);
}
else
{
  values[0] = pow(2*(1 - x[0]), k);
}
}
};"""

#------------------------------------------------------------------------------

def get_convergence_rates(norms):
  """
Convenience function. norms[0] is mesh.hmin();
norms[1], norms[2] is H10 and L2 norm from assemble
norms[3], [4], [5] is H10, L2, H-1 norm from eigenvalues
"""

  print "asseble norm convergence"
  N = len(norms)
  for i in range(1, N):
    for j, norm in zip([1, 2], ["H10", "L2"]):
      print norm, ln(norms[i, j]/norms[i-1, j])/ln(norms[i, 0]/norms[i-1, 0]),
    print

  print "\neigenvalue norm convergence"
  for i in range(1, N):
    for j, norm in zip([3, 4, 5], ["H10", "L2", "HM1"]):
      print norm, ln(norms[i, j]/norms[i-1, j])/ln(norms[i, 0]/norms[i-1, 0]),
    print

#------------------------------------------------------------------------------

