"""
Solve without fenics
-u``  = f in (0, 1)x(0, 1)  with f = 2*pi**2*sin(pi*x)*sin(pi*y)
u = 0 on boundary
"""
#TODO revisit this problem once the chapter on FEM assembly has been convered
#Right now we only show how a typical row in the stiffness matrix would look
#like and relate it to the 5 point discretization of laplacian with central
#differences in 2D

#TODO
# generate grid
# get connectivity
# assemble stiffness matrix
# assemble b
# assemble mass matrix for L2 norm
# get the rate

