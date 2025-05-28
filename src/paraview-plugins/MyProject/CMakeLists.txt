cmake_minimum_required(VERSION 3.18)
project(MyProject)

option(BUILD_SHARED_LIBS "Build shared libraries" ON)
find_package(VTK REQUIRED)
include(GNUInstallDirs)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
set(VTK_WRAP_PYTHON ON)

# Explicit install locations
set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Install path")
set(CMAKE_INSTALL_INCLUDEDIR "include")
set(CMAKE_INSTALL_LIBDIR "lib")
set(CMAKE_INSTALL_BINDIR "bin")
set(CMAKE_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake")

# This is what the error is complaining about
set(CMAKE_DESTINATION "${CMAKE_INSTALL_PREFIX}")


set(module_files
  "MyModule/vtk.module")

vtk_module_scan(
  MODULE_FILES ${module_files}
  PROVIDES_MODULES modules
  REQUIRES_MODULES required_modules
  UNRECOGNIZED_MODULES unrecognized_modules
  WANT_BY_DEFAULT ON)

vtk_module_build(
  MODULES ${modules})

# if (VTK_WRAP_PYTHON)
#   vtk_module_wrap_python(
#     MODULES             ${modules}
#     MODULE_DESTINATION  "site-packages"
#     PYTHON_PACKAGE      my_python_module
#   )
# endif ()

add_executable(MyStandalone main.cpp)
target_link_libraries(MyStandalone PRIVATE MyModule VTK::IOXML)
