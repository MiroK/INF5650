'''
Simple NS solver of pressure driven flow. Inflow and outflow pressures are
1 and 0. No slip boundary condition at top and bottom wall. Term grad(u).u is
linearized but Picard iteration is omitted.

`solver_type` decides if direct(MUMPS) or preconditioned iterative solver is
going to be used. Further `pressure_bc` chooses between enforcing pressure
boundary conditions weakly and strongly. The idea is to show that setting
the conditions strongly gives wrong result (a lot wrong).
'''

from dolfin import *

V_order = 2
solver_type = 'iterative'
pressure_bc = 'weak'
# direct, weak    : OK
# iterative, weak : OK
# direct, strong    : WRONG
# iterative, strong : WRONG

# Turn off the progress info
set_log_level(WARNING)

# Mesh specification
N = 32
mesh = UnitSquareMesh(N, N)

# Domain and values of no_slip bcs
def noslip_domain(x, on_boundary):
    return on_boundary and near(x[1]*(1-x[1]), 0)

noslip_value = Constant((0., 0.))

# FEM formulation with Taylor-Hood element
V = VectorFunctionSpace(mesh, 'CG', V_order)
Q = FunctionSpace(mesh, 'CG', 1)
W = MixedFunctionSpace([V, Q])

bc_noslip = DirichletBC(W.sub(0), noslip_value, noslip_domain)
bcs = [bc_noslip]

up = TrialFunction(W)
(u, p) = split(up)

vq = TestFunction(W)
(v, q) = split(vq)

# Initial conditions, 2 for velocity + 1 for pressure
up_ = interpolate(Expression(('0.0', '0.0', '0.0')), W)
u_, p_ = split(up_)

# Viscosity and time step
nu = Constant(1./8.)
dt = Constant(0.1)

# Setup things to be able to prescribe inflow pressure
n = FacetNormal(mesh)
p_io = Expression('1-x[0]')

class PressureBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0]*(1-x[0]), 0)

facet_f = FacetFunction('size_t', mesh, 0)
PressureBoundary().mark(facet_f, 1)
ds = Measure('ds')[facet_f]

# Mixed weak form, by parts used on viscosity term and pressure term
# Simple
if pressure_bc == 'weak':
    F = 1./dt*inner(u - u_, v)*dx + inner(dot(grad(u), u_), v)*dx +\
        nu*inner(grad(u), grad(v))*dx - p*div(v)*dx - q*div(u)*dx +\
        inner(p_io, dot(v, n))*ds(1)

elif pressure_bc == 'strong':
    F = 1./dt*inner(u - u_, v)*dx + inner(dot(grad(u), u_), v)*dx +\
        nu*inner(grad(u), grad(v))*dx - p*div(v)*dx - q*div(u)*dx
    # The pressure turn `killed` by test function

    # Add the pressure bc
    bc_pressure = DirichletBC(W.sub(1), p_io, facet_f, 1)
    bcs.append(bc_pressure)

a = lhs(F)
L = rhs(F)

# Allocate LA objects for the system
A = Matrix()
b = Vector()

if solver_type == 'direct':
    solver = LUSolver('mumps')
    solver.set_operator(A)

elif solver_type == 'iterative':
    # Preconditioner form
    precond = 1./dt*inner(u, v)*dx + nu*inner(grad(u), grad(v))*dx + p*q*dx

    P = Matrix()
    foo = Vector()

    solver = KrylovSolver('minres', 'amg')
    solver.set_operators(A, P)
    solver.parameters['monitor_convergence'] = True

# Time loop
t = 0
while t < 0.5:
    t += dt(0)
    # Assemble the main system
    assemble_system(a, L, bcs, exterior_facet_domains=facet_f,
                    A_tensor=A, b_tensor=b)

    # Assemble preconditioner
    if solver_type == 'iterative':
        assemble_system(precond, L, bcs, exterior_facet_domains=facet_f,
                        A_tensor=P, b_tensor=foo)

    solver.solve(up_.vector(), b)

# Plot the final velocity
u, p = up_.split(deepcopy=True)
plot(u)
plot(p)
interactive()

print 'Time step :', dt(0)
print 'Dofs used :', W.dim()
print 'Cells used :', mesh.num_cells()
print 'Functional :', u.vector().max()

# Functional look only at velocity. Check out pressure for P1-P1
