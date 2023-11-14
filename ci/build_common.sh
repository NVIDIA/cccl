#!/bin/bash

set -eo pipefail

# Ensure the script is being executed in its containing directory
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )";

# Script defaults
HOST_COMPILER=${CXX:-g++} # $CXX if set, otherwise `g++`
CXX_STANDARD=17
CUDA_COMPILER=${CUDACXX:-nvcc} # $CUDACXX if set, otherwise `nvcc`
CUDA_ARCHS= # Empty, use presets by default.

# Check if the correct number of arguments has been provided
function usage {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "The PARALLEL_LEVEL environment variable controls the amount of build parallelism. Default is the number of cores."
    echo
    echo "Options:"
    echo "  -v/--verbose: enable shell echo for debugging"
    echo "  -cuda: CUDA compiler (Defaults to \$CUDACXX if set, otherwise nvcc)"
    echo "  -cxx: Host compiler (Defaults to \$CXX if set, otherwise g++)"
    echo "  -std: CUDA/C++ standard (Defaults to 17)"
    echo "  -arch: Target CUDA arches, e.g. \"60-real;70;80-virtual\" (Defaults to value in presets file)"
    echo
    echo "Examples:"
    echo "  $ PARALLEL_LEVEL=8 $0"
    echo "  $ PARALLEL_LEVEL=8 $0 -cxx g++-9"
    echo "  $ $0 -cxx clang++-8"
    echo "  $ $0 -cxx g++-8 -std 14 -arch 80-real -v -cuda /usr/local/bin/nvcc"
    exit 1
}

# Parse options

# Copy the args into a temporary array, since we will modify them and
# the parent script may still need them.
args=("$@")
echo "Args: ${args[@]}"
while [ "${#args[@]}" -ne 0 ]; do
    case "${args[0]}" in
    -v | --verbose) VERBOSE=1; args=("${args[@]:1}");;
    -cxx)  HOST_COMPILER="${args[1]}"; args=("${args[@]:2}");;
    -std)  CXX_STANDARD="${args[1]}";  args=("${args[@]:2}");;
    -cuda) CUDA_COMPILER="${args[1]}"; args=("${args[@]:2}");;
    -arch) CUDA_ARCHS="${args[1]}";    args=("${args[@]:2}");;
    -disable-benchmarks) ENABLE_CUB_BENCHMARKS="false"; args=("${args[@]:1}");;
    -h | -help | --help) usage ;;
    *) echo "Unrecognized option: ${args[0]}"; usage ;;
    esac
done

# Convert to full paths:
HOST_COMPILER=$(which ${HOST_COMPILER})
CUDA_COMPILER=$(which ${CUDA_COMPILER})

GLOBAL_CMAKE_OPTIONS=""
if [[ -n "${CUDA_ARCHS}" ]]; then
    GLOBAL_CMAKE_OPTIONS+="-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS} "
fi

if [ $VERBOSE ]; then
    set -x
fi

# Begin processing unsets after option parsing
set -u

readonly PARALLEL_LEVEL=${PARALLEL_LEVEL:=$(nproc)}

if [ -z ${CCCL_BUILD_INFIX+x} ]; then
    CCCL_BUILD_INFIX=""
fi

# Presets will be configured in this directory:
BUILD_DIR="../build/${CCCL_BUILD_INFIX}"

# The most recent build will always be symlinked to cccl/build/latest
mkdir -p $BUILD_DIR
rm -f ../build/latest
ln -sf $BUILD_DIR ../build/latest

# Now that BUILD_DIR exists, use readlink to canonicalize the path:
BUILD_DIR=$(readlink -f "${BUILD_DIR}")

# Prepare environment for CMake:
export CMAKE_BUILD_PARALLEL_LEVEL="${PARALLEL_LEVEL}"
export CTEST_PARALLEL_LEVEL="1"
export CXX="${HOST_COMPILER}"
export CUDACXX="${CUDA_COMPILER}"
export CUDAHOSTCXX="${HOST_COMPILER}"

export CXX_STANDARD

# Print "ARG=${ARG}" for all args.
function print_var_values() {
    # Iterate through the arguments
    for var_name in "$@"; do
        if [ -z "$var_name" ]; then
            echo "Usage: print_var_values <variable_name1> <variable_name2> ..."
            return 1
        fi

        # Dereference the variable and print the result
        echo "$var_name=${!var_name:-(undefined)}"
    done
}

echo "========================================"
echo "pwd=$(pwd)"
print_var_values \
    BUILD_DIR \
    CXX_STANDARD \
    CXX \
    CUDACXX \
    CUDAHOSTCXX \
    NVCC_VERSION \
    CMAKE_BUILD_PARALLEL_LEVEL \
    CTEST_PARALLEL_LEVEL \
    CCCL_BUILD_INFIX \
    GLOBAL_CMAKE_OPTIONS
echo "========================================"
echo
echo "========================================"
echo "Current commit is:"
git log -1 || echo "Not a repository"
echo "========================================"
echo

function configure_preset()
{
    local BUILD_NAME=$1
    local PRESET=$2
    local CMAKE_OPTIONS=$3

    pushd .. > /dev/null

    cmake --preset=$PRESET --log-level=VERBOSE $GLOBAL_CMAKE_OPTIONS $CMAKE_OPTIONS
    echo "$BUILD_NAME configure complete."

    popd > /dev/null
}

function build_preset()
{
    local BUILD_NAME=$1
    local PRESET=$2

    source "./sccache_stats.sh" "start"
    pushd .. > /dev/null

    cmake --build --preset=$PRESET -v
    echo "$BUILD_NAME build complete."

    popd > /dev/null
    source "./sccache_stats.sh" "end"
}

function test_preset()
{
    local BUILD_NAME=$1
    local PRESET=$2

    pushd .. > /dev/null

    ctest --preset=$PRESET
    echo "$BUILD_NAME testing complete."

    popd > /dev/null
}

function configure_and_build_preset()
{
    local BUILD_NAME=$1
    local PRESET=$2
    local CMAKE_OPTIONS=$3

    configure_preset "$BUILD_NAME" "$PRESET" "$CMAKE_OPTIONS"
    build_preset "$BUILD_NAME" "$PRESET"
}
