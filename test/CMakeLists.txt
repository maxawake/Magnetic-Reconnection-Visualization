cmake_minimum_required(VERSION 3.18)
project(MyScanProject)

find_package(ParaView REQUIRED)
include(GNUInstallDirs)

paraview_plugin_scan(
  PLUGIN_FILES "${CMAKE_CURRENT_SOURCE_DIR}/ParallelVectors/paraview.plugin"
  PROVIDES_PLUGINS plugins
  ENABLE_BY_DEFAULT ON)

paraview_plugin_build(
  PLUGINS ${plugins}
)
