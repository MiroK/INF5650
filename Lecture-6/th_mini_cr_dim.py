'''
Make sure you understand why the spaces have the dimensions below.
'''

from dolfin import *

mesh = UnitSquareMesh(2, 2)

# Taylor-Hood
V = VectorFunctionSpace(mesh, 'CG', 2)
Q = FunctionSpace(mesh, 'CG', 1)
TH = MixedFunctionSpace([V, Q])
TH_dim = TH.dolfin_element().space_dimension()

print "Taylor-Hood P2-P1 in 2D has dim", TH_dim

# MINI
V = VectorFunctionSpace(mesh, 'CG', 1)
V = V + VectorFunctionSpace(mesh, 'Bubble', 3)
MINI = MixedFunctionSpace([V, Q])
MINI_dim = MINI.dolfin_element().space_dimension()

print "MINI in 2D has dim", MINI_dim

# Crouzeix-Raviart
V = VectorFunctionSpace(mesh, 'CR', 1)
Q = FunctionSpace(mesh, 'DG', 0)
CR = MixedFunctionSpace([V, Q])
CR_dim = CR.dolfin_element().space_dimension()

print "CR in 2D has dim", CR_dim





