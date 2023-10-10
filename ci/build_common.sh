#!/bin/bash

set -eo pipefail

# Ensure the script is being executed in its containing directory
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )";

# Script defaults
CUDA_COMPILER=${CMAKE_CUDA_COMPILER:-nvcc}

# Check if the correct number of arguments has been provided
function usage {
    echo "Usage: $0 [OPTIONS] <HOST_COMPILER> <CXX_STANDARD> <GPU_ARCHS>"
    echo "The PARALLEL_LEVEL environment variable controls the amount of build parallelism. Default is the number of cores."
    echo "The CMAKE_CUDA_COMPILER environment variable can be used to control the CUDA compiler. The -nvcc flag takes precedence."
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

if [ -z ${DEVCONTAINER_NAME+x} ]; then
    BUILD_DIR=../build/local
else
    BUILD_DIR=../build/${DEVCONTAINER_NAME}
fi

# The most recent build will always be symlinked to cccl/build/latest
mkdir -p $BUILD_DIR
rm -f ../build/latest
ln -sf $BUILD_DIR ../build/latest

COMMON_CMAKE_OPTIONS="
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=${CXX_STANDARD} \
    -DCMAKE_CUDA_STANDARD=${CXX_STANDARD} \
    -DCMAKE_CXX_COMPILER=${HOST_COMPILER} \
    -DCMAKE_CUDA_COMPILER=${CUDA_COMPILER} \
    -DCMAKE_CUDA_HOST_COMPILER=${HOST_COMPILER} \
    -DCMAKE_CUDA_ARCHITECTURES=${GPU_ARCHS} \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
"

echo "========================================"
echo "Begin build"
echo "pwd=$(pwd)"
echo "CUDA_COMPILER=$CUDA_COMPILER"
echo "HOST_COMPILER=$HOST_COMPILER"
echo "CXX_STANDARD=$CXX_STANDARD"
echo "GPU_ARCHS=$GPU_ARCHS"
echo "PARALLEL_LEVEL=$PARALLEL_LEVEL"
echo "BUILD_DIR=$BUILD_DIR"
echo "Current commit is:"
git log -1
echo "========================================"

function configure(){
    local CMAKE_OPTIONS=$1
    cmake -S .. -B $BUILD_DIR $COMMON_CMAKE_OPTIONS $CMAKE_OPTIONS -G Ninja
}

function build(){
    local BUILD_NAME=$1
    source "./sccache_stats.sh" start
    cmake --build $BUILD_DIR --parallel $PARALLEL_LEVEL -v
    echo "${BUILD_NAME} build complete"
    source "./sccache_stats.sh" end
}

function configure_and_build() {
    local BUILD_NAME=$1
    local CMAKE_OPTIONS=$2
    configure "$CMAKE_OPTIONS"
    build "$BUILD_NAME"
}
