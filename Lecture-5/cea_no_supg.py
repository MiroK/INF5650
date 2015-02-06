"""
Consider to convection diffusion problem

-alpha*u`` - u` = f in (0, 1)
u = g on \partial(0, 1)

Problem 1) f = alpha*pi**2*sin(pi*x) - pi*cos(pi*x), u(0) = u(1) = 0
Problem 2) f = 0, u(0) = 0, u(1) = 1

In both cases estimate the constant from Cea's lemma.
"""

from dolfin import *
import matplotlib.pyplot as plt
from math import sqrt, tanh, log

def problem_1(alpha_value, n_cell):
  """
  Solve problem 1 on mesh with n_cells for given a. Return mesh size,
  lower bound for constant and error of solution.
  """

  mesh = UnitIntervalMesh(n_cell)
  h = mesh.hmin()

  V = FunctionSpace(mesh, 'CG', 1)
  u = TrialFunction(V)
  v = TestFunction(V)

  alpha = Constant(alpha_value)
  f = Expression("alpha*pi*pi*sin(pi*x[0])-pi*cos(pi*x[0])", alpha=alpha_value)

  bc = DirichletBC(V, Constant(0), DomainBoundary())

  a = alpha*u.dx(0)*v.dx(0)*dx - u.dx(0)*v*dx
  L = f*v*dx

  u = Function(V)
  A, b = assemble_system(a, L, bc)
  solve(A, u.vector(), b)
  
  # for Cea's constant L we have norm(e, 1)/h/norm(u_exact, 2) <= L
  # the exact solution is u_exact = sin(pi*x) and so norm(u_exact, 2) is
  # L2 norm of D^2u ie pi**2*sqrt(2)/2

  u_exact = Expression("sin(pi*x[0])", degree=5)
  u_norm = pi**2*sqrt(2)/2.
  e_norm = errornorm(u_exact, u, 'h10')
  
  L = e_norm/h/u_norm

  return h, L, e_norm

#------------------------------------------------------------------------------

def problem_2(alpha_value, n_cell):
  """
  Solve problem 2 on mesh with n_cells for given a. Return mesh size,
  lower bound for constant and error of solution.
  """

  mesh = UnitIntervalMesh(n_cell)
  h = mesh.hmin()

  V = FunctionSpace(mesh, 'CG', 1)
  u = TrialFunction(V)
  v = TestFunction(V)

  alpha = Constant(alpha_value)
  f = Constant(0) 

  bc0 = DirichletBC(V, Constant(0), "near(x[0], 0)")
  bc1 = DirichletBC(V, Constant(1), "near(x[0], 1)")
  bcs = [bc0, bc1]

  a = alpha*u.dx(0)*v.dx(0)*dx - u.dx(0)*v*dx
  L = f*v*dx

  u = Function(V)
  A, b = assemble_system(a, L, bcs)
  solve(A, u.vector(), b)

  # for Cea's constant L we have norm(e, 1)/h/norm(u_exact, 2) <= L
  # the exact solution is u_exact = [exp(-x/a) - 1]/[exp(-1/a) - 1] 
  # and so norm(u_exact, 2) is L2 norm of D^2u,
  # sqrt(coth(1/2/a)/2/a**3)

  a = alpha_value

  u_exact = Expression("(exp(-x[0]/a)-1)/(exp(-1/a)-1)", a=a, degree=5)
  u_norm = sqrt(0.5/a**3/tanh(0.5/a))           # TODO make sure this is OK
  e_norm = errornorm(u_exact, u, 'h10')
  
  L = e_norm/h/u_norm

  return h, L, e_norm

#------------------------------------------------------------------------------

def process_results(problem, alpha_values, n_cells):
  """Run the problem for all aplhas and all meshes and create the plots."""
  plt.figure()
  for alpha_value in alpha_values:
    hs = []
    Ls = []

    h_, L, e_ = problem(alpha_value=alpha_value, n_cell=n_cells[0])
    hs.append(h_)
    Ls.append(L)

    for n_cell in n_cells[1:]:
      h, L, e = problem(alpha_value=alpha_value, n_cell=n_cell)
      hs.append(h)
      Ls.append(L)

      print "(%g), Rate for alpha=%.2E is %.2f" % (e, alpha_value, log(e/e_)/log(h/h_))

      h_ = h
      e_ = e

    mean_L=sum(Ls)/len(Ls)
    plt.semilogx(hs, Ls, label=r"$\alpha=%g, L=%.3f$" % (alpha_value, mean_L))
    print

  plt.legend(loc='best')
  plt.xlabel(r'$h$')
  plt.ylabel(r'$L$')
  plt.show()

#------------------------------------------------------------------------------

if __name__ == '__main__':
  alpha_values = [10**i for i in [0, -3, -6, -9]]
  n_cells = [2**i for i in [5, 7, 9, 11, 13]]

  # For the first problem the solution does not depend on alpha so the the 
  # neither should the lower bound.
  #process_results(problem_1, alpha_values, n_cells)
        
  # For the secon problem the solution depends on alpha and for small alpha
  # its is very large which should make the constant very small --> we know
  # that the constant should be positve and as the lower bound the program 
  # computes 0, so we didn't lear much new this way
  #process_results(problem_2, alpha_values, n_cells)

  # Note that the convergence rate of norm(u - u_h, H10) is about one as 
  # predicted by Cea + BH.

