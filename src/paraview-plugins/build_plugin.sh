#!/bin/sh

set -e

readonly workdir="/plugin"
readonly srcdir="$workdir/src"
readonly builddir="$workdir/build"

PARAVIEW_DIR=`find /builds/gitlab-kitware-sciviz-ci/build/install/lib/cmake -type d -regex "/builds/gitlab-kitware-sciviz-ci/build/install/lib/cmake/paraview-[0-9,.]*"`

rm -rf "$workdir"
mkdir -p "$builddir"
mkdir -p "$srcdir"
cd "$srcdir"
tar -xzvf /plugin.tgz
cd "$builddir"
cmake -C /plugin.cmake -DParaView_DIR=$PARAVIEW_DIR -DCMAKE_BUILD_TYPE=Release ../src
make -j $1
