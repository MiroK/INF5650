from dolfin import *
from numpy.random import random, choice


class RandomF(Expression):
    def eval(self, values, x):
        values[0] = random()


def make_random(n_cells, n_random):
    mesh = UnitIntervalMesh(n_cells)
    V = FunctionSpace(mesh, 'CG', 1)
    f = Function(V)

    indices = choice(V.dim(), n_random)
    values = random(n_random)
    f.vector()[indices] = values
    return f


def get_hm1_norm(f, n_cells):
    mesh = UnitIntervalMesh(n_cells)

    V = FunctionSpace(mesh, 'CG', 1)
    u = TrialFunction(V)
    v = TestFunction(V)

    # Form for H^1_0 norm
    a = inner(grad(u), grad(v))*dx
    # Form for L^2 inner product
    m = inner(u, v)*dx
    L = inner(Constant(0.), v)*dx

    bc = DirichletBC(V, Constant(0.), DomainBoundary())
    A, _ = assemble_system(a, L, bc)
    M, _ = assemble_system(m, L, bc)

    f_V = interpolate(f, V)
    F = f_V.vector()
    bc.apply(F)
    plot(f_V, interactive=True)

    g = Function(V)
    G = g.vector()

    A_dot_G = Function(V)
    A_DOT_G = A_dot_G.vector()

    M_dot_G = Function(V)
    M_DOT_G = M_dot_G.vector()

    functionals = []
    norms = []
    winner = (0, 0)
    # Compute the H^{-1 norm}
    hm1_norm = -1
    for i in range(1, V.dim()-1):
        # Get basis function of V
        G.zero()
        G[i] = 1

        A_DOT_G = A*G
        M_DOT_G = M*G

        g_norm = sqrt(A_DOT_G.inner(G))
        functional = abs(M_DOT_G.inner(F))

        value = functional/g_norm

        functionals.append(functional)
        norms.append(g_norm)

        if value > hm1_norm:
            hm1_norm = value
            winner = (functional, g_norm)

    print '(f, v) min max', min(functionals), max(functionals)
    print '(v, v)_1 min max ', min(norms), max(norms)
    print '(f_v), (v, v)_1 for norm', winner

    # Compute the L^{2} norm
    M_DOT_G = M*F
    l2_norm = sqrt(M_DOT_G.inner(F))

    # Compute the H^{1} norm
    A_DOT_G = A*F
    h1_norm = sqrt(A_DOT_G.inner(F))

    return h1_norm, l2_norm, hm1_norm

#------------------------------------------------------------------------------

# Notes
print '\033[1;37;32m%s\033[0m' % 'sin(k*pi*x)'
u = Expression('sin(k*pi*x[0])', k=1)
print '\033[1;37;34m\t%s %s\t%s\t%s\033[0m' %\
    ('k', 'H^{1} norm', 'L^{2} norm', 'H^{-1} norm')
for k in [1, 100, 100]:
    u.k = k
    h1_norm, l2_norm, hm1_norm = get_hm1_norm(u, n_cells=1000)
    print '\033[1;37;34m%10d %10E %10E %10E\033[0m'\
        % (k, h1_norm, l2_norm, hm1_norm)

print

# Random exercise
print '\033[1;37;32m%s\033[0m' % 'random'
u = RandomF()
print '\033[1;37;34m\t%s %s\t%s\t%s\033[0m' %\
    ('k', 'H^{1} norm', 'L^{2} norm', 'H^{-1} norm')
for k in [10, 100, 1000]:
    h1_norm, l2_norm, hm1_norm = get_hm1_norm(u, n_cells=k)
    print '\033[1;37;34m%10d %10E %10E %10E\033[0m'\
        % (k, h1_norm, l2_norm, hm1_norm)
