
"""
Consider to convection diffusion problem

-alpha*u`` - u` = f in (0, 1)
u = g on \partial(0, 1)

1) f = alpha*pi**2*sin(pi*x) - pi*cos(pi*x), u(0) = u(1) = 0
2) f = 0, u(0) = 0, u(1) = 1

Use supg stabilization and in both cases estimate the constant from
Cea's lemma. 
"""

# FIXME add problem 2
# FIXME add supg norm

from dolfin import *
import matplotlib.pyplot as plt
from math import sqrt, tanh
from cea_no_supg import process_results

def problem_1(alpha_value, n_cell):
  """
  Solve problem 1 on mesh with n_cells for given a. Return mesh size and
  lower bound for constant and error.
  """

  mesh = UnitIntervalMesh(n_cell)
  h = CellSize(mesh)

  V = FunctionSpace(mesh, 'CG', 1)
  u = TrialFunction(V)
  v = TestFunction(V)

  alpha = Constant(alpha_value)
  f = Expression("alpha*pi*pi*sin(pi*x[0])-pi*cos(pi*x[0])", alpha=alpha_value)
  velocity = Constant(-1.0)

  bc = DirichletBC(V, Constant(0), DomainBoundary())

  galerkin = alpha*u.dx(0)*v.dx(0)*dx + velocity*u.dx(0)*v*dx - f*v*dx 
  residuum = -div(alpha*grad(u)) + velocity*u.dx(0) - f        
  
  vnorm = sqrt(velocity*velocity)
  beta = 0.5/vnorm                     

  supg = h*beta*velocity*v.dx(0)*residuum*dx

  F = galerkin + supg

  a = lhs(F)
  L = rhs(F)

  u = Function(V)
  A, b = assemble_system(a, L, bc)
  solve(A, u.vector(), b)

  h = mesh.hmin() #!

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
  h = CellSize(mesh)

  V = FunctionSpace(mesh, 'CG', 1)
  u = TrialFunction(V)
  v = TestFunction(V)

  alpha = Constant(alpha_value)
  f = Constant(0) 

  bc0 = DirichletBC(V, Constant(0), "x[0] < DOLFIN_EPS")
  bc1 = DirichletBC(V, Constant(1), "x[0] > 1 - DOLFIN_EPS")
  bcs = [bc0, bc1]
  
  velocity = Constant(-1.0)
  galerkin = alpha*u.dx(0)*v.dx(0)*dx + velocity*u.dx(0)*v*dx - f*v*dx 
  residuum = -div(alpha*grad(u)) + velocity*u.dx(0) - f        
  
  vnorm = sqrt(velocity*velocity)
  beta = 0.5/vnorm                     

  supg = h*beta*velocity*v.dx(0)*residuum*dx

  F = galerkin + supg

  a = lhs(F)
  L = rhs(F)

  u = Function(V)
  A, b = assemble_system(a, L, bcs)
  solve(A, u.vector(), b)

  h = mesh.hmin() #!
  a = alpha_value

  u_exact = Expression("(exp(-x[0]/a)-1)/(exp(-1/a)-1)", a=a, degree=5)
  u_norm = sqrt(0.5/a**3/tanh(0.5/a))           # TODO make sure this is OK
  e_norm = errornorm(u_exact, u, 'h10')
  
  L = e_norm/h/u_norm

  plot(u**2, interactive=True, mesh=mesh)

  return h, L, e_norm

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------

if __name__ == '__main__':
  alpha_values = [10**i for i in [-9]]
  n_cells = [2**i for i in [5, 7, 9, 11, 13]]
  
  process_results(problem_2, alpha_values, n_cells)

