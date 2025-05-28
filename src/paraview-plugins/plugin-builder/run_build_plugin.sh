#!/bin/bash

# Options management
usage() { echo "Usage: $0 [-d <plugin pluginDir>] [-j jobs] hashOrTag

An open-source docker based script to facilitate the building of plugins for the binary release of ParaView.

Options:
  -d   Path to a directory containing a plugin. This directory should contain a CMakeLists.txt.
       If no directory is provided, this script will try to use a plugin.tgz file in the current directory.

  -j   Jobs : Number of parallel jobs to build the plugin with. Default is nproc+1.

hashOrTag:
  Must correspond to the tag of an existing kitware/paraview_org-plugin-devel dockerhub image

Notes:
  You might need to modify plugin.cmake in order to pass specific CMake options to your plugin." 1>&2; exit 1; }

let nbJobs=`nproc`+1
while getopts ":d:j:h" o; do
    case "${o}" in
        d)
            pluginDir=${OPTARG}
            ;;
        j)
            nbJobs=${OPTARG}
            ;;
        *)
            usage
            ;;
    esac
done
shift $((OPTIND-1))
hashOrTag=$1

if [ -z "${hashOrTag}" ]; then
    usage
fi

# Plugin archive management
deleteArchive=false
if ! [ -z ${pluginDir} ]
then
    if [ -d ${pluginDir} ]
    then
        echo ${pluginDir}
        # Create an archive from the plugin
        tar -czvf plugin.tgz -C ${pluginDir} .
        deleteArchive=true
    else
        echo "Provided plugin directory does not exist. Ignoring."
    fi
fi

if ! [[ -f plugin.tgz ]]
then
     echo "No plugin.tgz archive found. Aborting."; exit 0;
fi

# recover ParaView image
docker pull kitware/paraview_org-plugin-devel:$1

# Copy and build the plugin
docker create --name=paraview kitware/paraview_org-plugin-devel:$1 /bin/sh -c "while true; do echo waiting; sleep 1; done"
docker start paraview
docker cp build_plugin.sh paraview:/
docker cp plugin.cmake paraview:/
docker cp plugin.tgz paraview:/
echo "Building the plugin for ParaView ${hashOrTag} using ${nbJobs} jobs ..."
docker exec paraview scl enable devtoolset-7 -- sh /build_plugin.sh ${nbJobs}
docker cp paraview:/plugin/build ./output
docker stop paraview
docker rm paraview
echo "Done."
