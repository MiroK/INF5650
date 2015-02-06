
from l12_module import *
from numpy import zeros
from math import log as ln
from math import sqrt
import sympy as sym

# 1)
# I want to see what happens with eigenvalues if higher order spaces
# are used and meshes are refined.

#Ns = [10, 100, 1000]
#orders = [1, 2]

#eigenvalue_dependency_on_mesh_and_order(Ns, orders)

# Eigenvalues are very close.
# Higher order spaces mean larger eigenvalues (i.e f=eigenvanctions which
# oscillate more rapidly) but the same effect can be obtained if for given
# order finer mesh is used. Remember exact lambda~h**(-2). So, CG1 with fine
# mesh is sufficient.

#------------------------------------------------------------------------------

# 2)
# I want to know whether eigenvectors are orthonormal in L^2 norm as they
# should be.

#eigenvector_orthonormality_test(1000, 1)   # in P1
#! eigenvector_orthonormality_test(1000, 2)   # in P2 --- can't get Kent's code
                                        # to work for this scale, but see
                                        # qr scaling

# The eigenvectors are orthonormal.

#------------------------------------------------------------------------------

# 3)
# Exercise 2.1. The task is to compute norm of random function and
# observe the effect of meshes. Can it really be random? For the H1-seminorm to
# be computed, the function must be in H^1_0 (so that by parts integration is
# okay) so there should be some problems. Do these issues effect the other
#two norms?

# reproduce Kent's results for sin(k*pi*x)
#for k in [1, 10, 100]:
#  print "K IS", k
#  get_norms(Expression("sin(k*pi*x[0])", k=k), N=1000, order=1,
#            exact_solution=(k*pi/sqrt(2), 1./sqrt(2))) # OK
#
## What happens for abs(sin(k*pi*x))
#for k in [1, 10, 100]:
#  print "K IS", k
#  get_norms(Expression("fabs(sin(k*pi*x[0]))", k=k), N=1000, order=1,
#            exact_solution=(k*pi/sqrt(2), 1./sqrt(2))) # OK
#
#
#>----------------------------

Ns = [128, 256, 512, 1024]
norms = zeros((len(Ns), 6))

# See about mesh effect on sin(pi*x[0])
#k = 1
#
#for i, N in enumerate(Ns):
#  print "N IS", N
#  norms[i, : ] = get_norms(Expression("sin(k*pi*x[0])", k=k), N=N, order=1,
#                 exact_solution=(k*pi/sqrt(2), 1./sqrt(2)))

#get_convergence_rates(norms)

# For function in H10 well resolved by the cG1 and mesh, there is quadratic
# convergence in assemble norms and linear in eigenvalues

#>----------------------------

# Just to confirm the above, see about mesh effect on k*(x-0.5)**2 - 0.25*k
#K, X = sym.symbols("K X")
#F = K*(X - 0.5)**2 - K/4
#DF = sym.diff(F, X)
#
#l2 = sym.simplify(sym.integrate(F**2, (X, 0, 1)))
#h10 = sym.simplify(sym.integrate(DF**2, (X, 0, 1)))
#
#k = 10
#for i, N in enumerate(Ns):
#  print "N IS", N
#  h10_v = sqrt(h10.subs(K, k).evalf())
#  l2_v = sqrt(l2.subs(K, k).evalf())
#  norms[i, : ] = get_norms(Expression("k*(x[0]-0.5)*(x[0]-0.5) - 0.25*k", k=k),
#                 N=N, order=1, exact_solution=(h10_v, l2_v))
#
#get_convergence_rates(norms)

#>----------------------------

# See about function that is not H^1_0 because of bcs, f = k*x

#k = 1
#kx = Expression(kx_code)
#kx.k = k

#for i, N in enumerate(Ns):
#  print "N IS", N
#  norms[i, : ] = get_norms(kx, N=N, order=1, exact_solution=(k, k/sqrt(3)))

# In eigenvalues the l2 norm is okay, the basis functions can't resolve only
# the end point, which has measure 0
# H10 is wrong - we are ignoring the boundary term!

#<----------------------------------------------------------------------------

# Show that the above is in fact due to wrong bcs that cause function
# to be outside H10.

#kx0 = Expression(kx0_code)

#for i, N in enumerate(Ns):
#  print "N IS", N
#  kx0.h = 1./N
#  norms[i, : ] = get_norms(kx0, N=N, order=1,
#                          exact_solution=(None, None), with_plot=False)

#<-------------------------------------------------------------
# See about the function with a kink.

#k = 100
#
#kink = Expression(kink_code)
#kink.k = k
#
#for i, N in enumerate([1024]):
#  print "N IS", N
#  norms[i, : ] = get_norms(kink, N=N, order=1, exact_solution=(None),
#  with_plot=True)

#<--------------------------------------------------------------------

# 4) Exercise 2.5
# See about the hat function from exercise 2.5
#
#k = 1./100
#
#for i, N in enumerate([512]):
#  print "N IS", N
#  norms[i, : ] = get_norms(Hat(k), N=N, order=1,
#                           exact_solution=(sqrt(2./k), sqrt(2*k/3)), with_plot=True)

# 5) Random function with values in (0, 1)
#for i, N in enumerate([10, 100, 1000]):
#  norms[i, : ] = get_norms(None, N=N, order=1, exact_solution=None,\
#                           with_plot=True, random=True, abs_sin=False)
# 6) abs(sin(0.25*N*x[0])) a bit like random function with (0, 1)
#for i, N in enumerate([10, 100, 1000]):
#  norms[i, : ] = get_norms(None, N=N, order=1, exact_solution=None,\
#                           with_plot=True, random=False, abs_sin=True)


# What happens if:
# Run Kents code with N=1000 and sin(k*pi*x)
for k in []:#[1, 10, 100]:
  print "K IS", k
  get_norms(Expression("sin(k*pi*x[0])", k=k), N=1000, order=1,
            exact_solution=(k*pi/sqrt(2), 1./sqrt(2))) # OK

# Run Kents code with N=1000 and abs(sin(k*pi*x))
for k in []:#[1, 10, 100]:
  print "K IS", k
  get_norms(Expression("fabs(sin(k*pi*x[0]))", k=k), N=1000, order=1,
            exact_solution=(k*pi/sqrt(2), 1./sqrt(2))) # OK


# Run random type problem changing N and random values in (0, 1)
for i, N in enumerate([10, 100, 1000]):
  norms[i, : ] = get_norms(None, N=N, order=1, exact_solution=None,\
                           with_plot=True, random=1)
# Run random type problem changing N and random values in (-1, 1)
for i, N in enumerate([10, 100, 1000]):
  norms[i, : ] = get_norms(None, N=N, order=1, exact_solution=None,\
                           with_plot=True, random=-1)
# Run problem, changing N and also sin(0.25*N*x)
for i, N in enumerate([10, 100, 1000]):
  norms[i, : ] = get_norms(None, N=N, order=1, exact_solution=None,\
                           with_plot=True, sin=1)
# Run problem, changing N and also abs(sin(0.25*N*x))
for i, N in enumerate([10, 100, 1000]):
  norms[i, : ] = get_norms(None, N=N, order=1, exact_solution=None,\
                           with_plot=True, sin=-1)
