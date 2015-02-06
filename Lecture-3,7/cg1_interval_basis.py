"""
We want to draw the basis functions of finite element function space constructed
with linear continuous Lagrange elements on unit interval with m elements.
"""

from numpy import zeros, linspace, where
import matplotlib.pyplot as plt

m = 11            # number of elements
h = 1./m          # mesh size

#------------------------------------------------------------------------------

# Let the plot command do all the work
def psi(i, x):
  y = zeros(m)
  y[i] = 1
  return y

mesh = linspace(0, 1, m)
fig0 = plt.figure()
plt.plot(mesh, psi(0, mesh), label=r"$\psi_{0}$")
plt.plot(mesh, psi(5, mesh), label=r"$\psi_{5}$")
plt.plot(mesh, psi(10, mesh), label=r"$\psi_{10}$")
plt.legend(loc='best')

#------------------------------------------------------------------------------

# Do it yourself 1. Instead of mesh there will be [0, 1] with N points 
# and basis function actually need to be computed.

def phi(i, x):
  a = (i - 1)*h # support of i-th function is on two elements [a, c] and [c, b]
  c = i*h
  b = (i + 1)*h

  y = zeros(len(x))
  for j, x_ in enumerate(x):
    if a < x_ < b:
      if x_ < c:
        y[j] = (x_ - a)/h
      else:
        y[j] = (b - x_)/h
    else:
      y[j] = 0

  return y

x = linspace(0, 1, 100)
fig1 = plt.figure()
plt.plot(x, phi(0, x), label=r"$\phi_{0}$")
plt.plot(x, phi(5, x), label=r"$\phi_{5}$")
plt.plot(x, phi(10, x), label=r"$\phi_{10}$")
plt.legend(loc='best')

#------------------------------------------------------------------------------

# Do it yourself 2. Use the map from reference element and basis defined there.

def chi(i, x):
  def ref_map(x, L, R):
    return 2*(x - (L+R)/2)/(R-L)

  def base0(Y):
    return (1-Y)/2.

  def base1(Y):
    return (1+Y)/2.

  a = (i - 1)*h 
  c = i*h
  b = (i + 1)*h

  y = zeros(len(x))
  for j, x_ in enumerate(x):
    if a < x_ < b:
      if x_ < c:
        y[j] = base1(ref_map(x_, a, c))
      else:
        y[j] = base0(ref_map(x_, c, b))
    else:
      y[j] = 0

  return y

x = linspace(0, 1, 100)
fig1 = plt.figure()
plt.plot(x, chi(0, x), label=r"$\chi_{0}$")
plt.plot(x, chi(5, x), label=r"$\chi_{5}$")
plt.plot(x, chi(10, x), label=r"$\chi_{10}$")
plt.legend(loc='best')

#------------------------------------------------------------------------------

plt.show()
