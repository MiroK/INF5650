def test_oct2py(backends):
  """Check for oct2py."""
  try:
    import oct2py as op
    backends.append("oct2py")
  except ImportError:
    print "Oct2py not found."
  

#------------------------------------------------------------------------------

def test_pytave(backends):
  """Check for pytave."""
  try:
    import pytave
    backends.append("pytave")
  except ImportError:
    print "Pytave not found."


#------------------------------------------------------------------------------

def test_slepc(backends):
  """Check for petsc + slepc."""
  from dolfin import has_linear_algebra_backend, has_slepc
  if has_linear_algebra_backend("PETSc") and has_slepc():
    backends.append("petsc")
  else:
    print "Petsc + slepc not found."
