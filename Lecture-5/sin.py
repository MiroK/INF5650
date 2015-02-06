"""
Let norm(u, m) = int_{0, 1} -laplace(u)**m, u dx. Compute norm(u, m) for
u(x) = sin(k*pi*x) and different m\in Z.
"""

from random import choice
from numpy import linspace, sqrt, pi
import matplotlib.pyplot as plt

def plot_norms(ms):
  k = linspace(1, 100, 50)
  linecolors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
  linemarkers = [',', 'o', 'v', '^', '<', '>', '*', '+', 'x']
  plt.figure()
  for m in ms:
    if m != int(m):
      raise ValueError('Only interger valued norms are allowed.')
    else:
      lc = choice(linecolors)
      lm = choice(linemarkers)
      norm = sqrt(2.)*(k*pi)**m
      plt.semilogy(k, norm, marker=lm, color=lc, label='%d' % m)
      
  plt.legend(loc='best')
  plt.xlabel('k')
  plt.ylabel('norm')
  plt.show()

#------------------------------------------------------------------------------

plot_norms([-2, -1, 0, 1, 2, 3])
