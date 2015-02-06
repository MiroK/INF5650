"""
-u`` = f in (0, 1)
U(0) = u(1) = 0

u = sin(k*pi*x); f = (k*pi)**2*sin(k*pi*x)

Get the convergence rates in H10, L2, L1, L\infty norms.
"""

from dolfin import *
from math import log as ln

set_log_level(WARNING)

u_exact = Expression("sin(k*pi*x[0])", k=1)
f = Expression("k*pi*k*pi*sin(k*pi*x[0])", k=1)

def poisson(n_elements, p_degree, norm, k):
  """
  Solve the above problem on mesh with n_elements V_h|T = polynomial of
  degree p_degree. Return norm ["H10", "L2", "L1", "Loo"].
  """

  mesh = UnitIntervalMesh(n_elements)
  V = FunctionSpace(mesh, "CG", p_degree)

  u = TrialFunction(V)
  v = TestFunction(V)

  f.k = k
  u_exact.k = k
  a = inner(grad(u), grad(v))*dx
  L = f*v*dx
  bc = DirichletBC(V, Constant(0), DomainBoundary())

  u = Function(V)
  solve(a == L, u, bc)

  h = mesh.hmin()

  if norm == "H10":
    return h, errornorm(u_exact, u, "H10", degree_rise=4)
  
  elif norm == "L2":
    return h, errornorm(u_exact, u, "L2", degree_rise=4)
  
  else:
    u_ = interpolate(u_exact, V)
    u.vector()[:] -= u_.vector()
    u.vector().abs()
    
    if norm == "Loo":
      return h, u.vector().max()
    
    elif norm == "L1":
      return h, u.vector().sum()*h    # exact for p=1

if __name__ == "__main__":
  p = 1
  for k in [1, 10, 100]:
    print k, "-----------------------------------------------------------"
    for norm_type in ["H10", "L2", "Loo", "L1"]:
      h_, norm_ = poisson(n_elements=128, p_degree=p, norm=norm_type, k=k)
      print norm_type, "rate :",
      for i in [8, 9, 10, 11, 12, 13, 14, 15]:
        n = 2**i
        h, norm = poisson(n_elements=n, p_degree=p, norm=norm_type, k=k)
        print "%.2f" % (ln(norm/norm_)/ln(h/h_)), 
        #C = norm/(k*pi)**(p+1)/h**p,
        #print C, 
        #print norm, C[0]*(k*pi)**(p+1)/h**p
        h_, norm_ = h, norm
      print 
    print 

  # p=1 gives quadratic in all but H10 which is linear, no dependence on k
  # p=2; need k = 100 to get stable rates, then H10 is quadratic and the
  #      and the rest is more or less cubic
  # p=2; need k = 100 again, then H10 cubic, the others are order 4

  # are the oscill in rates due to error being small?
