'''
Convince yourself that FEM discretizations that don't comply with
inf-sup condition produce bad results.
'''

from dolfin import *
from termcolor import colored # sudo pip install termcolor
from math import isnan

# define domain for DirichletBC
def not_outlet(x, on_boundary):
  return on_boundary and (x[0] < DOLFIN_EPS or near(x[1]*(1 - x[1]), 0))

#------------------------------------------------------------------------------

def poisseuile(element, bubble=False, with_plot=False):
  '''Solve poisseuile flow with discretization given by element. Use bubbles
  optionaly and plot the results.'''
  mesh= UnitSquareMesh(64, 64)

  V = VectorFunctionSpace(mesh, *element['u']) 
    
  if bubble:
    V = V + VectorFunctionSpace(mesh,'Bubble',3)
        
  Q = FunctionSpace(mesh, *element['p'])
  M = MixedFunctionSpace([V, Q])

  u, p = TrialFunctions(M)
  v, q = TestFunctions(M)

  f = Constant((0,0))

  u_analytical = Expression(('x[1]*(1-x[1])', '0.0'))
  p_analytical = Expression('-2+2*x[0]')

  bc = DirichletBC(M.sub(0), u_analytical, not_outlet)

  a = inner(grad(u), grad(v))*dx + div(u)*q*dx + div(v)*p*dx
  L = inner(f, v)*dx

  up = Function(M)
  A,b = assemble_system(a, L, bc)
  solve(A, up.vector(), b)    #LU is default

  u, p = up.split(True)
    
  u_error = errornorm(u_analytical, u, norm_type='H10')
  p_error = errornorm(p_analytical, p, norm_type='L2')

  if with_plot:
    plot(u, title='velocity', interactive=True)
    plot(p, title='pressure', interactive=True)

  return u_error, p_error

#------------------------------------------------------------------------------

# create a dictionary with all tested elements
elements = {}
elements['th21'] = {'u' : ['CG', 2], 'p' : ['CG', 1]}
elements['th10'] = {'u' : ['CG', 1], 'p' : ['DG', 0]} # no inf sup
elements['th22'] = {'u' : ['CG', 2], 'p' : ['CG', 2]} # no inf sup
elements['th11'] = {'u' : ['CG', 1], 'p' : ['CG', 1]} # no inf sup
elements['th20'] = {'u' : ['CG', 2], 'p' : ['DG', 0]} 
elements['cr'] =   {'u' : ['CR', 1], 'p' : ['DG', 0]} # ok, but non-conforming
elements['mini'] = {'u' : ['CG', 1], 'p' : ['CG', 1]}

with_plot = False

for name, element in elements.items():
  if name == 'mini':
    u_error, p_error = poisseuile(element, with_plot=with_plot, bubble=True)
  else:
    u_error, p_error = poisseuile(element, with_plot=with_plot)

  if isnan(u_error) or isnan(p_error):
    inf_sup = False
  else:
    inf_sup = True

  print colored("%5s" % name.upper(), 'green' if inf_sup else 'red'),
  print "error in u = %.8g, error in p = %.8g" % (u_error, p_error)
