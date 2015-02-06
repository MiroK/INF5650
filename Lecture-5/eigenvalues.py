from dolfin import *
import sys
from numpy import zeros, arange
import matplotlib.pyplot as plt

'''
Solve the eigenvalue problem

alpha*u`` + v*u` = lmbda u in (0, 1)

where alpha is a diffusion constant, v is an advection velocity. Homogenous
Dirichlet boundary conditions are prescribed based on the order of the
differential operators considered.
'''

class Left(SubDomain):
  """Domain x[0] = 0."""
  def inside(self, x, on_boundary):
    return on_boundary and near(x[0], 0)

#------------------------------------------------------------------------------

class Right(SubDomain):
  """Domain x[0] = 1."""
  def inside(self, x, on_boundary):
    return on_boundary and near(x[0], 1)

#------------------------------------------------------------------------------

class EigenSolver():
  """Wrapper for PETScEigenSolver of the problem above + plotting."""
  def __init__(self, problem):
    """ 
      Problem is a dictionary with keywords alpha, v, bc_domain specifying
      respectively the diffusion constant, string for Expression that
      defines advection velocity and domain specifying where bcs are prescribed.
    """
    alpha = problem['alpha']
    velocity = Expression(problem['v'])
    bc_domain = problem['bc_domain']

    self.title = r"$\alpha=%3.E,\,v=%s,\, bc=%s$" % (alpha, velocity.cppcode,\
                                                     bc_domain)

    mesh = UnitIntervalMesh(500)
    V = FunctionSpace(mesh, "Lagrange", 1)
    self.V = V

    u = TrialFunction(V)
    v = TestFunction(V)

    self.a = Constant(alpha)*u.dx(0)*v.dx(0)*dx()
    self.a += velocity*u.dx(0)*v*dx()
    self.m = u*v*dx
    self.L = Constant(0)*v*dx

    if bc_domain == 'l':
      self.bc = DirichletBC(V, Constant(0), Left())
    elif bc_domain == 'r':
      self.bc = DirichletBC(V, Constant(0), Right())
    elif bc_domain == 'lr':
      self.bc = DirichletBC(V, Constant(0), DomainBoundary())
#------------------------------------------------------------------------------
  def solve(self, method=None, plot_eigenvectors_of=None):
    """
      Solve the generalized eigenproblem. Optional method chooses solution
      method in eigensolver. Number plot_eigenvectors_of will turn on plotting
      of eigenvectors of given eigenvalue.
    """
    a, m, L, bc, V = self.a, self.m, self.L, self.bc, self.V

    A = PETScMatrix()
    M = PETScMatrix()
    assemble_system(a, L, bc, A_tensor=A)
    assemble_system(m, L, bc, A_tensor=M)
    
    eigensolver = SLEPcEigenSolver(A, M)
    
    if method is not None:
      eigensolver.parameters["solver"] = method 

    timer = Timer("Eigensolver")
    print "Computing eigenvalues ...",
    sys.stdout.flush()
    timer.start()
    eigensolver.solve()
    timer.stop()
    print "done in", timing("Eigensolver")
   
    N = V.dim()
    n_pairs = eigensolver.get_number_converged()
    if n_pairs != N:
      raise AssertionError("Eigensolver did find %d of %d eigenpairs" %\
      (n_pairs, N)) 

    r_eigs = zeros(N) # real and imaginary part of eigenvalues
    i_eigs = zeros(N)

    real = Function(V)
    imag = Function(V)
    for i in range(N):
      r, c, rx, cx = eigensolver.get_eigenpair(i)
      r_eigs[i], i_eigs[i] = r, c
     
      if plot_eigenvectors_of is not None:
        if near(r, plot_eigenvectors_of):
          real.vector()[:] = rx
          imag.vector()[:] = cx
          plot(real, title='real', interactive=True)
          plot(imag, title='imag', interactive=True)

    self.r_eigs = r_eigs[:]
    self.i_eigs = i_eigs[:]

    print "Real part of spectrum in [%g, %g]" % (r_eigs.min(), r_eigs.max())
    print "Imaginary part of spectrum in [%g, %g]" % (i_eigs.max(), i_eigs.min())
#------------------------------------------------------------------------------

  def plot(self):
    indices = arange(self.V.dim())
    plt.figure()
    plt.subplot(311)
    plt.title(self.title)
    plt.plot(indices, self.r_eigs, 'r.')
    plt.ylabel(r'$\Re\lambda$')
    plt.subplot(312)
    plt.plot(indices, self.i_eigs, 'b*')
    plt.ylabel(r'$\Im\lambda$')
    plt.subplot(313)
    plt.plot(self.r_eigs, self.i_eigs, 'gx')
    plt.xlabel(r'$\Re\lambda$')
    plt.ylabel(r'$\Im\lambda$')

#------------------------------------------------------------------------------

# L1(u)=u_x. How do eigs change with boundary conditions?
params = {'alpha' : 0, 'v' : '1', 'bc_domain' : None}
for bc_domain in ['l', 'r', 'lr']:
  params['bc_domain'] = bc_domain
  L1 = EigenSolver(params)
  L1.solve(plot_eigenvectors_of=1)
  L1.plot()

# L2(u)=-alpha*u_xx. How do eigs change with alpha?
#params = {'alpha' : None, 'v' : '0', 'bc_domain' : 'lr'}
#for alpha in [1, 1E-3, 1E-6]:
#  params['alpha'] = alpha
#  L2 = EigenSolver(params)
#  L2.solve()
#  L2.plot()

# L3 = L1 + L2. How do eigs change with alpha
# This is incompressible case.
#params = {'alpha' : None, 'v' : '1', 'bc_domain' : 'lr'}
#for alpha in [1, 1E-3, 1E-6]:
#  params['alpha'] = alpha
#  L3 = EigenSolver(params)
#  L3.solve("lapack")
#  L3.plot()
# We have theorem that in this case Re(lambda) > 0


# L4(u) = x*u_x. How do eigs change with boundary conditions?
#params = {'alpha' : 0, 'v' : 'x[0]', 'bc_domain' : None}
#for bc_domain in ['l', 'r', 'lr']:
#  params['bc_domain'] = bc_domain
#  L4 = EigenSolver(params)
#  L4.solve("lapack")
#  L4.plot()

# L5 = L4 + L2. How do eigs change with alpha?
# This is compressible.
#params = {'alpha' : None, 'v' : 'x[0]', 'bc_domain' : 'lr'}
#for alpha in [1, 1E-3, 1E-6, 1E-9, 1E-12]:
#  params['alpha'] = alpha
#  L5 = EigenSolver(params)
#  L5.solve("lapack")
#  L5.plot()

# L6(u) = -xL1(u) + L2(u). How do eigs change with alpha?
#params = {'alpha' : None, 'v' : '-x[0]', 'bc_domain' : 'lr'}
#for alpha in [1, 1E-3, 1E-6, 1E-9, 1E-12]:
#  params['alpha'] = alpha
#  L5 = EigenSolver(params)
#  L5.solve("lapack")
#  L5.plot()
# Still incompressible but velocity = -x --> velocity` < 0 and we have a theorem
# that Re(lambda) > 0. 


plt.show()






