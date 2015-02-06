"""
We want to compute H^k(0, 1) norm of sin(m*pi*x).
"""

from math import pi
import matplotlib.pyplot as plt
from numpy import linspace, sqrt

def norm(m, k):
  return sqrt(0.5*((m*pi)**(2*(k + 1)) - 1)/((m*pi)**2 - 1))

m = linspace(1, 100) # continuous range of wave numbers

fig = plt.figure()

for k in [0, 1, 2, 3]:
  plt.semilogy(m, norm(m, k), label="%d" % k)

plt.xlabel(r"$m$")
plt.ylabel(r"$||\sin(m\pi x)||_{H^k(0, 1)}$")
plt.legend(loc='best')
plt.show()

# Note that for k=0, i.e. L^2 norm, the norm is insensitivie to wavenumbers
# m. This in this norm you don't see the oscillations. With k in the H^k
# norm the sensitivity to oscillations grows.
