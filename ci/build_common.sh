#!/bin/bash

set -eo pipefail

# Ensure the script is being executed in its containing directory
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )";

# Script defaults
CUDA_COMPILER=nvcc

# Check if the correct number of arguments has been provided
function usage {
    echo "Usage: $0 [OPTIONS] <HOST_COMPILER> <CXX_STANDARD> <GPU_ARCHS>"
    echo "The PARALLEL_LEVEL environment variable controls the amount of build parallelism. Default is the number of cores."
    echo "Example: PARALLEL_LEVEL=8 $0 g++-8 14 \"70\" "
    echo "Example: $0 clang++-8 17 \"70;75;80-virtual\" "
    echo "Possible options: "
    echo "  -nvcc: path/to/nvcc"
    echo "  -v/--verbose: enable shell echo for debugging"
    exit 1
}

# Check for extra options
# While there are more than 3 arguments, parse switches/options
while [ "$#" -gt 3 ]
do
  case "${1}" in
  -h)     usage ;;
  -help)  usage ;;
  --help) usage ;;
  --verbose)           VERBOSE=1; shift ;;
  -v)                  VERBOSE=1; shift ;;
  -nvcc)               CUDA_COMPILER="${2}"; shift 2;;
  -disable-benchmarks) ENABLE_CUB_BENCHMARKS="false"; shift ;;
  *) usage ;;
  esac
done

if [ $VERBOSE ]; then
    set -x
fi

if [ "$#" -ne 3 ]; then
    echo "Invalid number of arguments"
    usage
fi

# Begin processing unsets after option parsing
set -u

# Assign command line arguments to variables
readonly HOST_COMPILER=$(which $1)
readonly CXX_STANDARD=$2

# Replace spaces, commas and semicolons with semicolons for CMake list
readonly GPU_ARCHS=$(echo $3 | tr ' ,' ';')

readonly PARALLEL_LEVEL=${PARALLEL_LEVEL:=$(nproc)}
readonly NVCC_VERSION=$($CUDA_COMPILER --version | grep release | awk '{print $6}' | cut -c2-)

if [ -z ${DEVCONTAINER_NAME+x} ]; then
    CCCL_BUILD_INFIX=local
else
    CCCL_BUILD_INFIX=${DEVCONTAINER_NAME}
fi

# Presets will be configured in this directory:
BUILD_DIR=../build/${CCCL_BUILD_INFIX}

# The most recent build will always be symlinked to cccl/build/latest
mkdir -p $BUILD_DIR
rm -f ../build/latest
ln -sf $BUILD_DIR ../build/latest

# Prepare environment for CMake:
export CCCL_BUILD_INFIX
export CMAKE_BUILD_PARALLEL_LEVEL="${PARALLEL_LEVEL}"
export CMAKE_EXPORT_COMPILE_COMMANDS=1
export CTEST_PARALLEL_LEVEL="${PARALLEL_LEVEL}"
export CUDAARCHS="${GPU_ARCHS}"
export CUDACXX="${CUDA_COMPILER}"
export CUDAHOSTCXX="${HOST_COMPILER}"
export CXX="${HOST_COMPILER}"

echo "========================================"
echo "Begin build"
echo "pwd=$(pwd)"
echo "BUILD_DIR=$BUILD_DIR"
echo "CXX_STANDARD=$CXX_STANDARD"
echo "CXX=$CXX"
echo "CUDACXX=$CUDACXX"
echo "CUDAHOSTCXX=$CUDAHOSTCXX"
echo "CUDAARCHS=$CUDAARCHS"
echo "NVCC_VERSION=$NVCC_VERSION"
echo "CMAKE_BUILD_PARALLEL_LEVEL=$CMAKE_BUILD_PARALLEL_LEVEL"
echo "CTEST_PARALLEL_LEVEL=$CTEST_PARALLEL_LEVEL"
echo "CCCL_BUILD_INFIX=$CCCL_BUILD_INFIX"
echo "========================================"

function configure_preset()
{
    local BUILD_NAME=$1
    local PRESET=$2
    local CMAKE_OPTIONS=$3

    pushd .. > /dev/null

    cmake --preset=$PRESET $CMAKE_OPTIONS --log-level=VERBOSE
    echo "$BUILD_NAME configure complete."

    popd > /dev/null
}

function build_preset()
{
    local BUILD_NAME=$1
    local PRESET=$2

    source "./sccache_stats.sh" "start"
    pushd .. > /dev/null

    cmake --build --preset=$PRESET
    echo "$BUILD_NAME build complete."

    popd > /dev/null
    source "./sccache_stats.sh" "end"
}

function configure_and_build_preset()
{
    local BUILD_NAME=$1
    local PRESET=$2
    local CMAKE_OPTIONS=$3

    configure_preset "$BUILD_NAME" "$PRESET" "$CMAKE_OPTIONS"
    build_preset "$BUILD_NAME" "$PRESET"
}
