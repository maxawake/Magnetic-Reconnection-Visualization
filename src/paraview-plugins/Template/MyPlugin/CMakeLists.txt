paraview_add_plugin(MyFilter
  REQUIRED_ON_CLIENT
  REQUIRED_ON_SERVER
  VERSION "1.0"
  MODULES MyFilterModules
  MODULE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/MyFilterModule/vtk.module"
  SERVER_MANAGER_XML "MyPlugin.xml")
