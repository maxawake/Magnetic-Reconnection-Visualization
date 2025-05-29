set(classes
  vtkBifurcationLine
)

vtk_module_add_module(BifurcationLineModule
  CLASSES ${classes}
  FORCE_STATIC
  SOURCES vtkBifurcationLine.cxx
  HEADERS vtkBifurcationLine.h)
