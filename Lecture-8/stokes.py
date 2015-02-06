'''
This script investigates optimal-order preconditioner for the Stokes problem.
'''

import matplotlib.pyplot as plt
from dolfin import *
import numpy

# Exact solutions
u_exact = Expression(('sin(pi*x[1])', 'cos(pi*x[0])'))
p_exact = Expression('sin(2*pi*x[0])')
f = Expression(('pi*pi*sin(pi*x[1]) - 2*pi*cos(2*pi*x[0])',
                'pi*pi*cos(pi*x[0])'))


# Boundary conditions for velocity will be set on top and bottom
def top_bottom(x, on_boundary):
    return near(x[1]*(1 - x[1]), 0) and on_boundary


def stokes_solver(N, start_random=False):
    mesh = UnitSquareMesh(N, N)
    h = mesh.hmin()

    # Variational formulation
    V = VectorFunctionSpace(mesh, 'CG', 2)
    Q = FunctionSpace(mesh, 'CG', 1)
    M = MixedFunctionSpace([V, Q])

    up = TrialFunction(M)
    vq = TestFunction(M)
    u, p = split(up)
    v, q = split(vq)

    # The solution on left and right boundaries is such that
    # (v, grad(u).n)*ds + (p, v.n)*ds = 0 where ds is left and right boundaries
    # This property + DirichletBC on velocity make the weak form correct
    a = inner(grad(u), grad(v))*dx + div(v)*p*dx + q*div(u)*dx
    L = inner(f, v)*dx
    bcs = DirichletBC(M.sub(0), u_exact, top_bottom)

    # Form for use in constructing preconditioner matrix
    b = inner(grad(u), grad(v))*dx + p*q*dx

    # Assemble system
    A, bb = assemble_system(a, L, bcs)

    # Assemble preconditioner system
    P, btmp = assemble_system(b, L, bcs)

    # Create Krylov solver and AMG preconditioner
    solver = KrylovSolver('tfqmr', 'amg')
    solver.parameters['nonzero_initial_guess'] = True
    solver.parameters['relative_tolerance'] = 1.0e-8
    solver.parameters['absolute_tolerance'] = 1.0e-8
    solver.parameters['monitor_convergence'] = False
    solver.parameters['report'] = False
    solver.parameters['maximum_iterations'] = 50000

    # Associate operator (A) and preconditioner matrix (P)
    solver.set_operators(A, P)

    # Solve
    up = Function(M)
    if start_random:
        up.vector()[:] = numpy.random.rand(M.dim())
        print up.vector().sum()
    n_iters = solver.solve(up.vector(), bb)
    u, p = up.split()

    # Get the L2 norm of velocity, to check that we're getting correct
    # solution and just same number of iters
    u_error = errornorm(u_exact, u)

    dim = A.size(0)
    return h, u_error, n_iters, dim

# -----------------------------------------------------------------------------

plt.figure()
for start_random in [True, False]:
    print 'h\t\trate_L2\t#iters'
    h_, e_, n_iters, dim = stokes_solver(8)
    dims = []
    iters = []
    for N in [64, 96, 128, 192, 256, 328]:
        h, e, n_iters, dim = stokes_solver(N, start_random)

        rate_e = ln(e/e_)/ln(h/h_)

        print '%.2E\t%.2f\t%d' % (h, rate_e, n_iters)
        dims.append(dim)
        iters.append(n_iters)

    plt.plot(dims, iters, label=str(start_random))
plt.legend()
plt.show()
