"""
Solve without fenics
-u`` + u = f in (0, 1)   with f = sin(5*pi*x)
u(0) = u(1) = 0
"""
from math import log, sqrt
from numpy import sin, zeros, linspace, pi, cos, outer, trace
from scipy.linalg import toeplitz, solve
from scipy import interpolate
import matplotlib.pyplot as plt

# we use CG1 and unit interval partitioned uniformly into m elelements

def stiffness_matrix(m):
  col = zeros(m)
  col[0] = 2; col[1] = -1
  return toeplitz(col)

#------------------------------------------------------------------------------

def mass_matrix(m):
  col = zeros(m)
  col[0] = 4; col[1] = 1
  return toeplitz(col)

#------------------------------------------------------------------------------

def mixed_matrix(m):  # int phi_i*phi_j`
  matrix = zeros((m, m))
 
  for i in range(m - 1):
    matrix[i, i+1] = 0.5

  for i in range(1, m):
    matrix[i, i-1] = -0.5
   
  return matrix

#------------------------------------------------------------------------------

def poisson(m):
  h = 1./m
  x = linspace(h, 1-h, m-1)           # dof for x_0 and x_m are zero due bcs

  b = sin(5*pi*x)*h                   # intergrated by midpoint rule
  A = stiffness_matrix(m-1)/h    
  M = h*mass_matrix(m-1)/6
  AM = mixed_matrix(m-1)

  u_h = solve(A + M, b)

  u = sin(5*pi*x)/(25*pi**2 + 1)

  # FIXME is this correct?
  # u-u_h is expanded into basis so then l2^2 = int(dot(U_i*phi_i, Uh_j*phi_j)
  # which becomes U_i*Uh_j * int(phi_i, phi_j); i.e contraction of two matrices
  # U_Uh and the mass matrix
  U_Uh = outer((u_h - u).T, u_h - u)
  l2 = sqrt(trace(U_Uh*M)) # A:B = tr(A^T B) but A=A^T

  # get the h1 by representing u` as c_i*phi_` ... will get stiffness matrix
  h1 = sqrt(trace(U_Uh*A))

  # what if we wanted to use the knowledge of exact derivative and somehow
  # compute the derivative from the u_h solution, say by interpolation
  # the representation is then via phi_i
  #du = 5*pi*cos(5*pi*x)/(25*pi**2 + 1)
  #du_h = interpolate.splev(x, interpolate.splrep(x, u_h, s=0), der=1)
  #DU_DUh = outer((du - du_h).T, du - du_h)
  #h1 = sqrt(trace(DU_DUh*M))

  # quadratic in h1!; superconvergence
  
  return h, l2, h1

#-------------------------------------------------------------------------------

# compute convergence rates, l2 should be quadratic and h1 linear
h_, l2_, h1_ = poisson(32)
print "l2 rate | h1 rate"
for m in [64, 128, 256, 512, 1024]:
  h, l2, h1 = poisson(m)
  print "%7.2f" % (log(l2/l2_)/log(h/h_)),
  print "%7.2f" % (log(h1/h1_)/log(h/h_))
  h_, l2_, h1_ = h, l2, h1
