'''
Investigate the order of approximation of wall shear streass obtained with
various Taylor-Hood discretizations of the Stokes problem
'''

from dolfin import *
from math import log as ln
from math import sqrt as msqrt

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
  u_analytical = Expression(('sin(pi*x[1])','cos(pi*x[0])'), degree=7)

  bc = DirichletBC(M.sub(0), u_analytical, not_outlet)

  a = inner(grad(u), grad(v))*dx + div(u)*q*dx + div(v)*p*dx
  L = inner(f, v)*dx

  up = Function(M)
  A, b = assemble_system(a, L, bc)
  solve(A, up.vector(), b)

  u, p = up.split(True)

  # Get error of velocity in H10, the idea is that there might be
  # a link between convergence rate of u and shear
  error_velocity = errornorm(u_analytical, u, 'H10')

  # Compute wall shear stress error as its L2(boundary) norm
  order = V.ufl_element().degree() + 3
  DG = VectorFunctionSpace(mesh, 'DG', order)
  u = interpolate(u, DG)
  u_analytical = interpolate(u_analytical, DG)

  e = Function(DG)
  e.assign(u)
  e.vector().axpy(-1.0, u_analytical.vector())

  n = FacetNormal(mesh)
  t = as_vector([-n[1], n[0]])
  #L2 norm
  error_shear = assemble((2*dot(t, dot(sym(grad(e)), n)))**2*ds)
  error_shear = sqrt(error_shear)

  #L1 norm
  #error_shear = assemble(abs(2*dot(t, dot(sym(grad(e)), n)))*ds)

  #TODO Loo norm

  h = mesh.hmin()
  return h, error_shear, error_velocity

#------------------------------------------------------------------------------

elements = {}
elements['th43'] = {'u' : ['CG', 4], 'p' : ['CG', 3]}
elements['th42'] = {'u' : ['CG', 4], 'p' : ['DG', 2]}
elements['th32'] = {'u' : ['CG', 3], 'p' : ['CG', 2]}
elements['th31'] = {'u' : ['CG', 3], 'p' : ['CG', 1]}
elements['th21'] = {'u' : ['CG', 2], 'p' : ['CG', 1]}

for name, element in elements.items():
  h_, es_, ev_ = problem(n_cells=8, element=element)
  print 'Element:', name
  for n_cells in [16, 32, 64]:
    h, es, ev = problem(n_cells=n_cells, element=element)

    rates = ln(es/es_)/ln(h/h_)
    ratev = ln(ev/ev_)/ln(h/h_)
    print '\tRate shear(L2(boundary)) = %.2f' % rates,
    print 'Rate velocity(H10) = %.2f' % ratev

#       SHEAR     VEL
# th43 |  4    |  4
# th42 |  3    |  3
# th32 |  3    |  3
# th31 |  2    |  2
# th21 |  2    |  2

# same
