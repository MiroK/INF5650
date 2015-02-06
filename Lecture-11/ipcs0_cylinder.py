'''
IPCS0(simple explicit treatment of nonlinear term) solver for cylinder flow.
'''

from dolfin import *

V_order = 1
print 'Solving IPCS0 with %d order of velocity space' % V_order

# Turn off the progress info
set_log_level(WARNING)

# Create the mesh
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

class OutflowBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], x_max)

# FEM form
V = VectorFunctionSpace(mesh, 'CG', V_order)
Q = FunctionSpace(mesh, 'CG', 1)

# Define test and trial functions
u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)

# Previous time step
u0 = Function(V)
p0 = Function(Q)

# Second to last time step
u1 = Function(V)
p1 = Function(Q)

# Constants
nu = Constant(1./1000)
dt = Constant(0.001)

# Tentative velocity, solve to u1
U = 0.5*(u + u0)
F0 = (1./dt)*inner(u - u0, v)*dx + inner(dot(grad(u0), u0), v)*dx\
        + nu*inner(grad(U), grad(v))*dx + inner(grad(p0), v)*dx
a0, L0 = system(F0)

# Projection, solve to p1
F1 = inner(grad(p - p0), grad(q))*dx + (1./dt)*q*div(u1)*dx
a1, L1 = system(F1)

# Finalize, solve to u1
F2 = (1./dt)*inner(u - u1, v)*dx + inner(grad(p1 - p0), v)*dx
a2, L2 = system(F2)

# Assemble matrices
A0 = assemble(a0)
A1 = assemble(a1)
A2 = assemble(a2)

# Create solvers
# Solver02 for tentative and finalize
solver02 = KrylovSolver()

# Solver1 for projection
solver1 = KrylovSolver()

# Prepare for boundary conditions
noslip_domain = NoslipBoundary()
inflow_domain = InflowBoundary()
outflow_domain = OutflowBoundary()

noslip_value = Constant((0., 0.))
inflow_value = \
    Expression(('4*Um*(x[1]*(ymax-x[1]))*sin(pi*t/8.0)/(ymax*ymax)',
                '0.0'), Um=1.5, ymax=y_max, t=0)

bc_noslip = DirichletBC(V, noslip_value, noslip_domain)
bc_outflow = DirichletBC(Q, Constant(0.), outflow_domain)
bcs_p = [bc_outflow]

# Time loop
t = 0
while t < 8:
    t += dt(0)
    inflow_value.t = t
    bc_inflow = DirichletBC(V, inflow_value, inflow_domain)
    bcs_u = [bc_noslip, bc_inflow]

    b = assemble(L0)
    [bc.apply(A0, b) for bc in bcs_u]
    solver02.solve(A0, u1.vector(), b)

    b = assemble(L1)
    [bc.apply(A1, b) for bc in bcs_p]
    solver1.solve(A1, p1.vector(), b)

    b = assemble(L2)
    [bc.apply(A2, b) for bc in bcs_u]
    solver02.solve(A2, u1.vector(), b)

    # Update
    p0.assign(p1)
    u0.assign(u1)

u_pvd = File('u.pvd')
u_pvd << u0

p_pvd = File('p.pvd')
p_pvd << p0

front = Point(c_x - r - DOLFIN_EPS, c_y)
back = Point(c_x + r + DOLFIN_EPS, c_y)

print 'Time step :', dt(0)
print 'Dofs used :', V.dim() + Q.dim()
print 'Cells used :', mesh.num_cells()
print 'Mesh hmin :', mesh.hmin()
print 'Functional :', p0(front) - p0(back)
