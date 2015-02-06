"""
Solve
-laplace(u) + v.grad(u) = f in [0, 1]
                      u = g on inflow boundary
             -grad(u).n = h on outflow boundary
"""

from dolfin import *
from numpy import zeros

compute_eigenvalues=True       # turn off for large meshes

mesh = UnitSquareMesh(25, 25)

# mark the inflow (1) and out flow parts(2) of the mesh boundary, based on the
# velocity field v = (1, 1),
# that is, left, bottom are in 
# that is right and top edges are out, 

mesh_boundary = FacetFunction("size_t", mesh, 0)
D= mesh.topology().dim()
mesh.init(D-1, D)
for facet in facets(mesh):
  M = facet.midpoint()
  x = M.x()
  y = M.y()

  if facet.exterior():
    if (near(x, 0) or near(y, 0)):      
      mesh_boundary[facet] = 1
    else:
      mesh_boundary[facet] = 2

#plot(mesh_boundary, interactive=True, title="mesh boundary") # just checking

V = FunctionSpace(mesh, 'CG', 1)
u = TrialFunction(V)
phi = TestFunction(V)

m = u*phi*dx                # for eigenvalue computations

f = Constant(1.)
g = Expression("cos(pi*x[0])*cos(pi*x[1])")
h = Constant(0)     
v = Expression(("a*x[0] + b", "c"), a=0, b=10, c=10)

bc = DirichletBC(V, g, mesh_boundary, 1)
ds = Measure("ds")[mesh_boundary]
diffusion = inner(grad(u), grad(phi))*dx

# first consider v = (1, 1), i.e. incompressible, h = 0, proper bcs
#convection = inner(v, grad(u))*phi*dx
#L = f*phi*dx - h*phi*ds(2)
#> the bc value is taken with the stream and diffusis along the way
#> increase velocity makes no boundary layer

#------------------------------------------------------------------------------

# change the neumann conditions to inhomegenous
#convection = inner(v, grad(u))*phi*dx
#h = Expression("2*cos(pi*x[0])*x[1]")
#L = f*phi*dx - h*phi*ds(2)
#> this only changes rhs, so nothing bad happens, but as it should the solution
#> is diffrent

#------------------------------------------------------------------------------

# incompressible v = (-1, -1), but keep old boundary tags, so that
# now inflow has neumann and outflow has dirichlet
#v.b = -1
#v.c = -1
#convection = inner(v, grad(u))*phi*dx
#L = f*phi*dx - h*phi*ds(2)
#> it is now the neumann cond, that is taken by the flow into the domain
#> and then hits derichlet, if speed big enough you get oscillations
#> also see what happens with spectum at high velocities

#------------------------------------------------------------------------------

# change flow to compressible (10*x[0], 1), h = 0, bcs int/out flow are correct
h = Expression("2*cos(pi*x[0])*x[1]")
v.a = 1E8      # try 1E8
v.b = 0
v.c = 1
convection = inner(v, grad(u))*phi*dx
L = f*phi*dx - h*phi*ds(2)
#> The higher v.a the more the solution is dictates by bc at y-axis. The speed
#> also effects the spectrum but even at 1E8 I was not able to make the form 
#> not positive definite. TODO maybe different flow pattern

#------------------------------------------------------------------------------

a = diffusion + convection

u = Function(V)
A = PETScMatrix()
b = Vector()

assemble_system(a, L, bc, exterior_facet_domains=mesh_boundary, A_tensor=A,\
                b_tensor=b)
solve(A, u.vector(), b)

plot(u, interactive=True)
print "Minima of u", u.vector().min()
print "Neumann term h*ds(2) evaluates to", assemble(h*ds(2), mesh=mesh)

#------------------------------------------------------------------------------

if compute_eigenvalues:
  M = PETScMatrix()
  assemble_system(m, L, bc, A_tensor=M, exterior_facet_domains=mesh_boundary)

  import matplotlib.pyplot as plt

  eigensolver = SLEPcEigenSolver(A, M)
  eigensolver.parameters["solver"] = "lapack" 

  # Compute all eigenvalues of A x = \lambda x
  print "\nComputing eigenvalues. This can take a minute."
  eigensolver.solve()

  N = V.dim()
  r_eigs = zeros(N)
  i_eigs = zeros(N)
  for i in range(N):
    r, c, rx, cx = eigensolver.get_eigenpair(i)
    r_eigs[i], i_eigs[i] = r, c
    if near(c, 1E-13) and near(r, 1E-13) and r > 0:
      print "[%g, %g]" % (r, c), "possible 0 in the spectrum"

  plt.figure()

  rmax, imax = r_eigs[0], i_eigs[0]
  rmin, imin = r_eigs[-1], i_eigs[-1]

  print "Largest real %g and complex i*%g eigenvalues" % (rmax, imax)
  print "Smallest real %g and complex i*%g eigenvalues" % (rmin, imin)
  
  plt.subplot(211)
  plt.plot(r_eigs, ".", label=r"$\Re{\lambda},\,\lambda_{min}=%g$" % rmin)
  plt.legend(loc="best")
  
  plt.subplot(212)
  plt.plot(i_eigs, ".", label=r"$\Im{\lambda},\,\lambda_{min}=%g$" % imin)
  plt.legend(loc="best")
  
  plt.show()

