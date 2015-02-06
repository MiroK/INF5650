'''
Test convergence rate of various Taylor-Hood discretizations for the
Stokes problem
'''

from dolfin import *
from math import log as ln

# define domain for DirichletBC
def not_outlet(x, on_boundary):
  return on_boundary and (x[0] < DOLFIN_EPS or near(x[1]*(1 - x[1]), 0))

#------------------------------------------------------------------------------

def problem(n_cells, element):
  '''Solve Stokes problem using V_h constructed from element on n_cells x
  n_cells mesh.'''

  mesh = UnitSquareMesh(n_cells, n_cells)
  V = VectorFunctionSpace(mesh, *element['u']) 
  Q = FunctionSpace(mesh, *element['p'])
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

  up = Function(M)
  A, b = assemble_system(a, L, bc)
  solve(A, up.vector(), b)

  u, p = up.split(True)
  
  u_error = errornorm(u_analytical, u, norm_type='H10')
  p_error = errornorm(p_analytical, p, norm_type='L2')

  h = mesh.hmin()
  return h, u_error + p_error, M.dim()

#------------------------------------------------------------------------------

elements = {}
elements['th43'] = {'u' : ['CG', 4], 'p' : ['CG', 3]}
elements['th42'] = {'u' : ['CG', 4], 'p' : ['DG', 2]} 
elements['th32'] = {'u' : ['CG', 3], 'p' : ['CG', 2]} 
elements['th31'] = {'u' : ['CG', 3], 'p' : ['CG', 1]} 

for name, element in elements.items():
  h_, e_, dim = problem(n_cells=8, element=element)
  print 'Element:', name
  for n_cells in [16, 32, 64]:
    h, e, dim = problem(n_cells=n_cells, element=element)
  
    rate = ln(e/e_)/ln(h/h_)
    print '\t(problem size %dx%d) Rate %.2f' %(dim, dim, rate)
