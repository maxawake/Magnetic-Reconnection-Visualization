set(interfaces)
set(sources
  vtkBifurcationLineMSR.cxx
  vtkBifurcationLineMSR.h
)

# paraview_plugin_add_auto_start(
#   CLASS_NAME vtkBifirucationLineMSR
#   STARTUP onStartup
#   SHUTDOWN onShutdown
#   INTERFACES autostart_interface
#   SOURCES autostart_sources)
# list(APPEND interfaces
#   ${autostart_interface})
# list(APPEND sources
#   ${autostart_sources})

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "/opt/paraview/lib")

paraview_add_plugin(BifurcationLine
  REQUIRED_ON_CLIENT
  VERSION "1.0"
  UI_INTERFACES ${interfaces}
  SOURCES ${sources}
  SERVER_MANAGER_XML
    BifurcationLineMSR.xml
  )

# target_link_libraries(BifurcationLine
#   PRIVATE
#     ParaView::pqApplicationComponents)

# option(BUILD_TESTING "Build Plugin Testing" OFF)
# if (BUILD_TESTING AND BUILD_SHARED_LIBS)
#   add_subdirectory(Testing)
# endif()
