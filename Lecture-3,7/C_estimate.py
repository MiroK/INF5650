"""
-u`` = f in (0, 1)
U(0) = u(1) = 0

u = sin(k*pi*x); f = (k*pi)**2*sin(k*pi*x)

On CG1 estimate the interpolation constant in H10 norm.
"""

from rates import *

for k in [1, 10, 100]:
  rhs_norm = (k*pi)**2/sqrt(2)           # L2 norm of D^{2}sin(k*pi*x)  

  print k, "-----------------------------------------------------------"
  for i in [8, 9, 10, 11, 12, 13, 14, 15, 16]:
    n = 2**i
    h, norm = poisson(n_elements=n, p_degree=1, norm="H10", k=k)
    C = norm/rhs_norm/h   # equal both sides in Bramble-Hilbert
    print "C=", C
  print 
print 

# C = 0.288675
# wolfram says that this is close to sqrt(3)/6
# TODO It would be nice to compute the constant for P1 just as in the notes
