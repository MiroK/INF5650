'''
Simple NS solver for flow past a cylinder with prescibed velocity on inflow,
no slip on the channel walls and `do nothing` on the outflow. Nonlinear term
is linearized but the Picard loop is skipped.
'''

from dolfin import *

V_order = 2

# Turn off the progress info
set_log_level(WARNING)

# geometric parameters
x_min, x_max = 0, 2.2
y_min, y_max = 0, 0.41          # note that cylinder is a bit off center
c_x, c_y, r = 0.2, 0.2, 0.05

def refine_cylinder(mesh):
  'Refine mesh by cutting cells around the cylinder.'
  h = mesh.hmin()
  center = Point(c_x, c_y)
  cell_f = CellFunction('bool', mesh, False)
  for cell in cells(mesh):
    if cell.midpoint().distance(center) < r + h:
      cell_f[cell] = True
  mesh = refine(mesh, cell_f)

  return mesh

# Create the mesh
# First define the domain
rect = Rectangle(x_min, y_min, x_max, y_max)
circ = Circle(c_x, c_y, r)
domain = rect - circ

# Mesh the domain
mesh = Mesh(domain, 45)

# Refine mesh n-times
n = 1
for i in range(n):
  mesh = refine_cylinder(mesh)

# Define domains
class InflowBoundary(SubDomain):
  def inside(self, x, on_boundary):
    return on_boundary and near(x[0], x_min)

class NoslipBoundary(SubDomain):
  def inside(self, x, on_boundary):
    dx = x[0] - c_x
    dy = x[1] - c_y
    dr = sqrt(dx**2 + dy**2)
    return on_boundary and (near(x[1]*(y_max - x[1]), 0) or dr < r + 1E-3)

# FEM formulation with Taylor-Hood element
V = VectorFunctionSpace(mesh, 'CG', V_order)
Q = FunctionSpace(mesh, 'CG', 1)
W = MixedFunctionSpace([V, Q])

# Prepare boundary conditions
noslip_domain = NoslipBoundary()
inflow_domain = InflowBoundary()

noslip_value = Constant((0., 0.))
inflow_value = \
    Expression(('4*Um*(x[1]*(ymax-x[1]))*sin(pi*t/8.0)/(ymax*ymax)',
                '0.0'), Um=1.5, ymax=y_max, t=0)

bc_noslip = DirichletBC(W.sub(0), noslip_value, noslip_domain)
bc_inflow = DirichletBC(W.sub(0), inflow_value, inflow_domain)
bcs = [bc_inflow, bc_noslip]


up = TrialFunction(W)
(u, p) = split(up)

vq = TestFunction(W)
(v, q) = split(vq)

# Initial conditions, 2 for velocity + 1 for pressure
up_ = interpolate(Expression(('0.0', '0.0', '0.0')), W)
u_, p_ = split(up_)

# Viscosity and time step
nu = Constant(1./1000)
dt = Constant(0.005)

# Mixed weak form, by parts used on viscosity term and pressure term
# Simple
F = 1./dt*inner(u - u_, v)*dx + inner(dot(grad(u), u_), v)*dx +\
    nu*inner(grad(u), grad(v))*dx - p*div(v)*dx - q*div(u)*dx

a = lhs(F)
L = rhs(F)

# Time loop
t = 0
while t < 8:
    t += dt(0)
    inflow_value.t = t
    solve(a == L, up_, bcs)

# Plot the final velocity
u, p = up_.split(deepcopy=True)

u_pvd = File('u.pvd')
u_pvd << u

p_pvd = File('p.pvd')
p_pvd << p

front = Point(c_x - r - DOLFIN_EPS, c_y)
back = Point(c_x + r + DOLFIN_EPS, c_y)

print 'Time step :', dt(0)
print 'Dofs used :', W.dim()
print 'Cells used :', mesh.num_cells()
print 'Mesh hmin :', mesh.hmin()
print 'Functional :', p(front) - p(back)
