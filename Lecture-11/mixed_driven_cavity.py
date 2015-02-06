'''
This script solves a driven cavity problem for Navier-Stokes equations. Mixed
formulation is employed and the time-discretization leads to fully non-linear
problem, which is handled internally by FEniCS.
'''


from dolfin import *

# Turn off the progress info
set_log_level(WARNING)

# Mesh specification
N = 32
mesh = UnitSquareMesh(N, N)

# Domain and values of no_slip bcs
def noslip_domain(x, on_boundary):
    return on_boundary and (near(x[1], 0) or near(x[0]*(1 - x[0]), 0))


noslip_value = Constant((0., 0.))

# Domain and values of lid
def driven_domain(x, on_boundary):
    return on_boundary and near(x[1], 1)

driven_value = Constant((1., 0.))

# FEM formulation with Taylor-Hood element
V = VectorFunctionSpace(mesh, 'CG', 2)
Q = FunctionSpace(mesh, 'CG', 1)
W = MixedFunctionSpace([V, Q])

bc_driven = DirichletBC(W.sub(0), driven_value, driven_domain)
bc_noslip = DirichletBC(W.sub(0), noslip_value, noslip_domain)
bcs = [bc_driven, bc_noslip]

up = Function(W)
(u, p) = split(up)

vq = TestFunction(W)
(v, q) = split(vq)

# Initial conditions, 2 for velocity + 1 for pressure
up_ = interpolate(Expression(('0.0', '0.0', '0.0')), W)
u_, p_ = split(up_)

# Viscosity and time step
mu = Constant(1E-2)
dt = Constant(0.005)

# Crank-Nicolson terms
U = 0.5*(u + u_)
P = 0.5*(p + p_)

# Mixed weak form, by parts used on viscosity term and pressure term
a = inner(u, v)*dx - inner(u_, v)*dx + dt*inner(dot(grad(U), U), v)*dx +\
    dt*mu*inner(grad(U), grad(v))*dx - dt*P*div(v)*dx -\
    dt*q*div(U)*dx

# Time loop
t = 0
while t < 0.25:
    t += dt(0)
    solve(a == 0, up, bcs,
          solver_parameters={'newton_solver': {'maximum_iterations':  25}})
    up_.assign(up)

# Plot the final velocity
plot(project(u, V))
interactive()
