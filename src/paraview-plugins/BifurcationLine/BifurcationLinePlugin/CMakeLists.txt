paraview_add_plugin(BifurcationLine
  REQUIRED_ON_CLIENT
  REQUIRED_ON_SERVER
  VERSION "1.0"
  MODULES BifurcationLineModule
  MODULE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/BifurcationLineModule/vtk.module"
  SERVER_MANAGER_XML "BifurcationLine.xml")
