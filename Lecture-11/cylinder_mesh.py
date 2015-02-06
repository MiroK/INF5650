'''
This script illustrates how to create a mesh for the cylinder test case with
the built-in mesh generation tools of DOLFIN. The mesh is created by meshing 
the domain, which is a difference of background rectangle and a circle. Moreover
the mesh around the cylinder is refined n-times. The script also shows how to
mark inflow, outflow and no-slip domains.
'''

from dolfin import *

# geometric parameters
x_min, x_max = 0, 2.2
y_min, y_max = 0, 0.41          # note that cylinder is a bit off center
c_x, c_y, r = 0.2, 0.2, 0.05

def refine_cylinder(mesh):
  'Refine mesh by cutting cells around the cylinder.'
  h = mesh.hmin()
  center = Point(c_x, c_y)
  cell_f = CellFunction('bool', mesh, False)
  for cell in cells(mesh):
    if cell.midpoint().distance(center) < r + h:
      cell_f[cell] = True
  mesh = refine(mesh, cell_f)

  return mesh

# Create the mesh
# First define the domain
rect = Rectangle(x_min, y_min, x_max, y_max)
circ = Circle(c_x, c_y, r)
domain = rect - circ

# Mesh the domain
mesh = Mesh(domain, 25)

# Refine mesh n-times
n = 3
for i in range(n):
  mesh = refine_cylinder(mesh)

# Plot mesh
plot(mesh, interactive=True)

# Define domains
class InflowBoundary(SubDomain):
  def inside(self, x, on_boundary):
    return on_boundary and near(x[0], x_min)

class OutflowBoundary(SubDomain):
  def inside(self, x, on_boundary):
    return on_boundary and near(x[0], x_max)

class NoslipBoundary(SubDomain):
  def inside(self, x, on_boundary):
    dx = x[0] - c_x
    dy = x[1] - c_y
    dr = sqrt(dx**2 + dy**2)
    return on_boundary and (near(x[1]*(y_max - x[1]), 0) or dr < r + 1E-3)

# Create facet function and mark the surfaces
facet_f = FacetFunction('size_t', mesh, 0)
InflowBoundary().mark(facet_f, 5)    # 5 is inflow
OutflowBoundary().mark(facet_f,6)   # 6 is inflow
NoslipBoundary().mark(facet_f, 7)    # 7 is inflow

plot(facet_f, interactive=True)
