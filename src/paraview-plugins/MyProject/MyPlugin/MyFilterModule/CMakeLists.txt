set(classes
  vtkMyFilter
)

vtk_module_add_module(MyFilterModules
  CLASSES ${classes}
  FORCE_STATIC
  SOURCES vtkMyFilter.cxx
  HEADERS vtkMyFilter.h)
