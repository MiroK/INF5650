"""
For u = sin(k*pi*x) investigate H^m Sobolev norms on (0, 1) of (u-u_hp)
where u_hp is the interpolant of u constructed with continuous Lagrange 
elements with degree p and mesh wish size h.
"""

from dolfin import *
from numpy import array
from math import log as ln
from math import sqrt as msqrt
import matplotlib.pyplot as plt

def my_errornorm(u, u_hp, m_norm):
  """
  Compute the m-norm th seminorm of u-u_hp. Combines the implementations
  of norm and errornorm from FEniCS. No sanity checks for types, etc.
  """
  # compute the error by interpolating both u and u_hp onto higher order DG
  # space 
  mesh = u_hp.function_space().mesh()
  order = u_hp.ufl_element().degree() + 3

  if u.ufl_element().degree() < order:
    print "Increase order in the expresion!"
    exit()

  DG = FunctionSpace(mesh, 'DG', order)
  
  u = interpolate(u, DG)
  u_hp = interpolate(u_hp, DG)

  # Compute the difference
  e = Function(DG)
  e.assign(u)
  e.vector().axpy(-1.0, u_hp.vector())

  if m_norm == 0:
    norm = assemble(inner(e, e)*dx,\
                 form_compiler_parameters={"representation" : "quadrature"})
  elif m_norm == 1:
    norm = assemble(inner(grad(e), grad(e))*dx,\
                 form_compiler_parameters={"representation" : "quadrature"})
  elif m_norm == 2:
    norm = assemble(inner(div(grad(e)), div(grad(e)))*dx,\
                 form_compiler_parameters={"representation" : "quadrature"})
  else:
    raise ValueError("Only H_m norms, m=0, 1, 2 are suported.")

  if norm < 0:
    print "Warning. Round off problems?", norm
    norm = -1     # this is a flag indicator
  else:
    norm = msqrt(norm)

  return norm

#------------------------------------------------------------------------------

def get_interpolation_error(u, n_cells, p_order, m_norm):
  """
  Construct V on [0, 1] discretized into n_cells and use p_order polynomials. 
  Return m_norm (0, 1, 2) of the interpolation error of u and mesh size.
  """
  
  mesh = UnitSquareMesh(n_cells, n_cells)
  h = mesh.hmin()
  V = FunctionSpace(mesh, 'CG', p_order)

  u_hp = interpolate(u, V)

  error_norm = my_errornorm(u, u_hp, m_norm)
 
  flag = error_norm > 0        # is result valid
  return h, error_norm, flag

#------------------------------------------------------------------------------

class RatePlotter:
  """Plot convergence rates from meshsizes, errors, and polynm. order"""
  
  def __init__(self, u, np, m_norm):
    """
    Init np plots positioned in [n_rows, n_cols]. Argument u, m_norm is used 
    to get the title.
    """
    self.np = np
    self.counter = 0
    if np > 3:
      self.n_rows = 3
      self.n_cols = (np-1)/self.n_rows + 1
    else:
      self.n_cols = 1
      self.n_rows = np

    cppcode = u.cppcode.replace('pi', '\pi')
    cppcode = cppcode.replace('sin', '\sin')
    cppcode = cppcode.replace('k', str(int(u.k)))
    cppcode = cppcode.replace('[0]', '')
    cppcode = cppcode.replace('*', ' ')
    title = r"$P_{p}$ interpolation of $%s$ in $H^{%d}$ norm" % \
            (cppcode, m_norm)
    # the error measures H^m_norm seminorm but that is the one that determines
    # convergence rate 
    plt.figure()
    plt.title(title)

    self.m_norm = m_norm
  
  def plot(self, hs, es, p):
    "Plot the rates."""
    self.counter +=1
    
    hs = array(hs); es = array(es)
    theory_rate = p + 1 - self.m_norm

    plt.subplot(self.n_rows, self.n_cols, self.counter)
    plt.loglog(hs, es, "-*", label=r"measured for $P_{%d}$" % p)
    plt.loglog(hs, hs**theory_rate, label="theory, %d" % theory_rate)
    plt.xlabel(r'$h$')
    plt.ylabel(r'$e$')
    plt.legend(loc='best')

    if self.counter == self.np:
      plt.show()

#------------------------------------------------------------------------------

def get_convergence_rate(u, p_order, m_norm, with_plot, n_cells_max=8):
  """
  Run get_interpolation_error(u, ..., p_order, m_norm) on series of meshes
  with 2**i cells, 1<i<n_cells_max, and print the convergence rates.
  Plot optionally.
  """

  if type(p_order) is not list:
    p_order = [p_order]

  if with_plot:
    n_plots = len(p_order)
    plotter = RatePlotter(u, n_plots, m_norm)

  for p in p_order:
    hs = []
    es = []

    h_, e_, flag = get_interpolation_error(u, 4, p, m_norm)
    hs.append(h_)
    es.append(e_)

    for n_cells in [2**i for i in range(3, n_cells_max)]:
      h, e, flag = get_interpolation_error(u, n_cells, p, m_norm)
      hs.append(h)
      es.append(e)
      
      if flag:
        rate = ln(e/e_)/ln(h/h_)
        print "(%10g) Rate of P_%d interpolant in H^%d seminorm is %.2f" % \
              (e, p, m_norm, rate)
      else:
        break

    print 
    
    if with_plot:
      plotter.plot(hs, es, p)

#------------------------------------------------------------------------------

u = Expression("sin(k*pi*x[0])", k=5, degree=9)    # should be safe for p_order
                                                   # less than 4

#IN THE GRAPGHS ITS ALL ABOUT MEASURED LINE AND THEORETICAL LINE BEING PARALLEL

#1 Check that for m_norm=0 we get rate of p_order+1
u.k = 1
#get_convergence_rate(u, p_order=[1, 2, 3, 4], m_norm=0, with_plot=True)
#> OK

#2 The last error is quite small so running this test on even finer
#meshes with k=1 could be effected by round-off errors. We'll therefore make
#the interpolated function more challenging by making the sine more oscillatory.
#Notice that this increases the size of the error but not the final rate if the
#meshes are fine enough to resolve the function.
#get_convergence_rate(u, p_order=[1, 2, 3, 4], m_norm=0, with_plot=True)
#> OK

#3 Check that for m_norm=1 we get rate of p_order, keep the challenging sine
#get_convergence_rate(u, p_order=[1, 2, 3, 4], m_norm=1, with_plot=True)
#> OK

#4 V_h that are from CG_p are only H^1 conforming so taking H^2 and higher order
# Sobolev spaces does not make sense. FEniCS however does not see the delta
# function that are in the D^2(u_hp) derivatives so he's able to come up with
# a result that agrees with the BH lemma. But you should reallize that this
# is due to way the intergral is computed!
# Lemma predict rate of p_order-1
get_convergence_rate(u, p_order=[1, 2, 3, 4], m_norm=2, with_plot=True)
#> OK, BUT SEE ABOVE
