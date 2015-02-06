'''
Test convergence rate of various stabilized P1-P1 discretization of the
Stokes problem
'''

from dolfin import *
from math import log as ln

# define domain for DirichletBC
def not_outlet(x, on_boundary):
  return on_boundary and (x[0] < DOLFIN_EPS or near(x[1]*(1 - x[1]), 0))

#------------------------------------------------------------------------------

def stabilized_problem(n_cells, method):
  '''Solve Stokes problem using P1-P1 stabilized elements on n_cells x
  n_cells mesh. Stabilization is either via MASS matrix of STIFFness matrix.'''

  mesh = UnitSquareMesh(n_cells, n_cells)
  V = VectorFunctionSpace(mesh, 'CG', 1) 
  Q = FunctionSpace(mesh, 'CG', 1)
  M = MixedFunctionSpace([V, Q])

  u, p = TrialFunctions(M)
  v, q = TestFunctions(M)

  f = Expression(('pi*pi*sin(pi*x[1])-2*pi*cos(2*pi*x[0])',\
                  'pi*pi*cos(pi*x[0])'))

  u_analytical = Expression(('sin(pi*x[1])','cos(pi*x[0])'))
  p_analytical = Expression('sin(2*pi*x[0])')
  
  bc = DirichletBC(M.sub(0), u_analytical, not_outlet)

  a = inner(grad(u), grad(v))*dx + div(u)*q*dx + div(v)*p*dx
  L = inner(f, v)*dx
  
  h = mesh.hmin()
  if method == 'MASS':
    epsilon = Constant(1E-2*h) # this is empirical
    a -= epsilon*inner(p, q)*dx
  elif method == 'STIFF':
    epsilon = Constant(1E-2*h**2) # and so is this
    a -= epsilon*inner(grad(p), grad(q))*dx

  up = Function(M)
  A, b = assemble_system(a, L, bc)
  solve(A, up.vector(), b)

  u, p = up.split(True)
  
  u_error = errornorm(u_analytical, u, norm_type='H10')
  p_error = errornorm(p_analytical, p, norm_type='L2')

  return h, u_error + p_error, M.dim()

#------------------------------------------------------------------------------

for method in ['MASS', 'STIFF']:
  h_, e_, dim = stabilized_problem(n_cells=8, method=method)
  print 'Stabilization', method
  for n_cells in [16, 32, 64, 128]:
    h, e, dim = stabilized_problem(n_cells=n_cells, method=method)

    rate = ln(e/e_)/ln(h/h_)
    print '\t(problem size %dx%d) Rate %.2f' %(dim, dim, rate)
