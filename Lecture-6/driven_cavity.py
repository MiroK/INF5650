from dolfin import *

'''
Demonstrate that the solution of the Stokes problem with Dirichlet bcs on
velocity everywhere on the boundary is effected by whether a direct or iterative
solver is used.

We consider driven cavity on [0, 1]^2 with u=(1, 0) at y=1 and no slip else.
'''

class Lid(SubDomain):
  '''Characteristic of lid.'''
  def inside(self, x, on_boundary):
    return on_boundary and (x[1] > 1 - DOLFIN_EPS)

#------------------------------------------------------------------------------

class Wall(SubDomain):
  '''Characteristic of boundary - lid.'''
  def inside(seld, x, on_boundary):
    return on_boundary and not (x[1] > 1 - DOLFIN_EPS)

#------------------------------------------------------------------------------

def driven_cavity(solver):
  '''Solve the driven cavity problem with iterative or direct solver.'''
  
  mesh = UnitSquareMesh(64, 64)

  # Taylor-Hood inf sup OK
  V = VectorFunctionSpace(mesh, 'CG', 2)
  Q = FunctionSpace(mesh, 'CG', 1)
  M = MixedFunctionSpace([V, Q])

  u, p = TrialFunctions(M)
  v, q = TestFunctions(M)

  a = inner(grad(u), grad(v))*dx + inner(div(v), p)*dx + inner(div(u), q)*dx
  L = inner(Constant((0., 0.)), v)*dx

  # mark the walls
  boundaries = FacetFunction('size_t', mesh, 0)
  Lid().mark(boundaries, 1)
  Wall().mark(boundaries, 2)
  #plot(boundaries)    # visual check of boundaries

  bc_lid = DirichletBC(M.sub(0), Constant((1., 0.)), boundaries, 1)
  bc_wall = DirichletBC(M.sub(0), Constant((0., 0.)), boundaries, 2)

  bcs = [bc_lid, bc_wall]

  up = Function(M)
  A, b = assemble_system(a, L, bcs)
 
  if solver == 'direct':
    solver = LUSolver()
    solver.solve(A, up.vector(), b)
  elif solver == 'iterative':
    solver = KrylovSolver()
    # try different krylov solver methods
    # default, cg, gmres, minres, richardson, bicgstab, tfqmgr
    #   no     no    no       no      no       no          no
    # gmres is default

    solver.parameters['monitor_convergence'] = True
    solver.parameters['maximum_iterations'] = 200
    solver.solve(A, up.vector(), b)

  u, p = up.split()

  plot(u, title='velocity')
  plot(p, title='pressure')
  interactive()

#------------------------------------------------------------------------------

#driven_cavity(solver='direct')
driven_cavity(solver='iterative')
