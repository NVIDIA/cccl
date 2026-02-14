#!/bin/bash

set -eo pipefail

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "This script must be sourced, not executed directly." >&2
  exit 1
fi

# Ensure the script is being executed in its containing directory
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )";

# Script defaults
VERBOSE=${VERBOSE:-}
HOST_COMPILER=${CXX:-g++} # $CXX if set, otherwise `g++`
CXX_STANDARD=17
CUDA_COMPILER=${CUDACXX:-nvcc} # $CUDACXX if set, otherwise `nvcc`
CUDA_ARCHS= # Empty, use presets by default.
GLOBAL_CMAKE_OPTIONS=()
DISABLE_CUB_BENCHMARKS= # Enable to force-disable building CUB benchmarks.
CONFIGURE_ONLY=false

# Check if the correct number of arguments has been provided
function usage {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "The PARALLEL_LEVEL environment variable controls the amount of build parallelism. Default is the number of cores minus one."
    echo
    echo "Options:"
    echo "  -v/-verbose: enable shell echo for debugging"
    echo "  -configure: Only run cmake to configure, do not build or test."
    echo "  -cuda: CUDA compiler (Defaults to \$CUDACXX if set, otherwise nvcc)"
    echo "  -cxx: Host compiler (Defaults to \$CXX if set, otherwise g++)"
    echo "  -std: CUDA/C++ standard (Defaults to 17)"
    echo "  -arch: Target CUDA arches, e.g. \"60-real;70;80-virtual\" (Defaults to value in presets file)"
    echo "  -cmake-options: Additional options to pass to CMake"
    echo
    echo "Examples:"
    echo "  $ PARALLEL_LEVEL=8 $0"
    echo "  $ PARALLEL_LEVEL=8 $0 -cxx g++-9"
    echo "  $ $0 -cxx clang++-8"
    echo "  $ $0 -configure -arch 80"
    echo "  $ $0 -cxx g++-8 -std 14 -arch 80-real -v -cuda /usr/local/bin/nvcc"
    echo "  $ $0 -cmake-options \"-DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS=-Wfatal-errors\""
    exit 1
}

# Check for required dependencies
function check_required_dependencies() {
    local missing_deps=()

    # Check for essential tools
    local required_tools=("cmake" "git" "ninja" "nproc")
    for tool in "${required_tools[@]}"; do
        command -v "$tool" &>/dev/null || missing_deps+=("$tool")
    done

    if [ ${#missing_deps[@]} -ne 0 ]; then
        echo "❌ Error: Missing required dependencies:" >&2
        printf "   • %s\n" "${missing_deps[@]}" >&2
        echo >&2
        exit 1
    fi
}

# Parse options

# Copy the args into a temporary array, since we will modify them and
# the parent script may still need them.
args=("$@")
while [ "${#args[@]}" -ne 0 ]; do
    case "${args[0]}" in
    -v | --verbose | -verbose) VERBOSE=1; args=("${args[@]:1}");;
    -configure) CONFIGURE_ONLY=true;   args=("${args[@]:1}");;
    -cxx)  HOST_COMPILER="${args[1]}"; args=("${args[@]:2}");;
    -std)  CXX_STANDARD="${args[1]}";  args=("${args[@]:2}");;
    -cuda) CUDA_COMPILER="${args[1]}"; args=("${args[@]:2}");;
    -arch) CUDA_ARCHS="${args[1]}";    args=("${args[@]:2}");;
    -disable-benchmarks) DISABLE_CUB_BENCHMARKS=1; args=("${args[@]:1}");;
    -cmake-options)
        if [ -n "${args[1]}" ]; then
            IFS=' ' read -ra split_args <<< "${args[1]}"
            GLOBAL_CMAKE_OPTIONS+=("${split_args[@]}")
            args=("${args[@]:2}")
        else
            echo "Error: No arguments provided for -cmake-options"
            usage
            exit 1
        fi
        ;;
    -h | -help | --help) usage ;;
    *) echo "Unrecognized option: ${args[0]}"; usage ;;
    esac
done

# Convert to full paths and validate compilers exist:
function validate_and_resolve_compiler() {
    local compiler_name="$1"
    local compiler_var="$2"
    local compiler_path

    compiler_path=$(which "${compiler_var}" 2>/dev/null)
    if [ -z "$compiler_path" ]; then
        echo "❌ Error: ${compiler_name} '${compiler_var}' not found in PATH" >&2
        exit 1
    fi

    echo "$compiler_path"
}

HOST_COMPILER=$(validate_and_resolve_compiler "Host compiler" "${HOST_COMPILER}")
CUDA_COMPILER=$(validate_and_resolve_compiler "CUDA compiler" "${CUDA_COMPILER}")

if [[ "$(basename "$CUDA_COMPILER")" == nvcc* ]]; then
    NVCC_VERSION=$("$CUDA_COMPILER" --version | grep "release" | sed 's/.*, V//')
    # Verify that we have an X.Y.Z version in case the output format changes:
    if ! [[ "$NVCC_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        echo "❌ Error: Detected nvcc version is not a valid X.Y.Z triple: '$NVCC_VERSION'" >&2
        echo "$CUDA_COMPILER --version" >&2 || :
        $CUDA_COMPILER --version >&2 || :
        exit 1
    fi
fi


if [[ -n "${CUDA_ARCHS}" ]]; then
    GLOBAL_CMAKE_OPTIONS+=("-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS}")
fi

if [ $VERBOSE ]; then
    set -x
fi

# Check for required dependencies
check_required_dependencies

# Begin processing unsets after option parsing
set -u

readonly PARALLEL_LEVEL=${PARALLEL_LEVEL:=$(nproc --all --ignore=1)}

if [ -z ${CCCL_BUILD_INFIX+x} ]; then
    CCCL_BUILD_INFIX=""
fi

mkdir -p ../build
# Absolute path to cccl/build
BUILD_ROOT=$(cd "../build" && pwd)

# Absolute path to per-devcontainer build directory
BUILD_DIR="$BUILD_ROOT/$CCCL_BUILD_INFIX"

# The most recent devcontainer build dir will always be symlinked to cccl/build/latest
mkdir -p $BUILD_DIR
rm -f $BUILD_ROOT/latest
ln -sf $BUILD_DIR $BUILD_ROOT/latest

# The more recent preset build dir will always be symlinked to:
# cccl/preset-latest
function symlink_latest_preset {
    local PRESET=$1
    mkdir -p "$BUILD_DIR/$PRESET"
    rm -f "$BUILD_ROOT/preset-latest"
    ln -sf "$BUILD_DIR/$PRESET" "$BUILD_ROOT/preset-latest"
}

# Now that BUILD_DIR exists, use readlink to canonicalize the path:
BUILD_DIR=$(readlink -f "${BUILD_DIR}")

# Prepare environment for CMake:
export CMAKE_BUILD_PARALLEL_LEVEL="${PARALLEL_LEVEL}"
export CTEST_PARALLEL_LEVEL="1"
export CXX="${HOST_COMPILER}"
export CUDACXX="${CUDA_COMPILER}"
export CUDAHOSTCXX="${HOST_COMPILER}"
export CXX_STANDARD

source ./pretty_printing.sh

# Kill any build / test steps that exceed this time, otherwise CI jobs may be
# killed by GHA before they can upload logs / artifacts needed to reproduce the timeout.
# Only applies when running inside GitHub Actions.
# Note that this is per-build/test limit, not a total timeout for the entire job.
: "${CCCL_CI_COMMAND_TIMEOUT:=5.5h}"

print_environment_details() {
  begin_group "⚙️ Environment Details"

  echo "free -h:"
  free -h || :

  echo "nproc=$(nproc || :)"

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
      CCCL_CI_COMMAND_TIMEOUT \
      CCCL_CUDA_EXTENDED \
      CCCL_BUILD_INFIX \
      GLOBAL_CMAKE_OPTIONS \
      TBB_ROOT

  echo "Current commit is:"
  git log -1 --format=short || echo "Not a repository"

  if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
  else
    echo "nvidia-smi not found"
  fi

  if command -v sccache &> /dev/null; then
    sccache --version
  else
    echo "sccache not found"
  fi

  if command -v cmake &> /dev/null; then
    cmake --version
  else
    echo "cmake not found"
  fi

  if command -v ctest &> /dev/null; then
    ctest --version
  else
    echo "ctest not found"
  fi

  end_group "⚙️ Environment Details"
}

run_ci_timed_command() {
    local group_name="${1:-}"
    shift
    local -a command=("$@")

    if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
        if [[ -n "${CCCL_CI_COMMAND_TIMEOUT}" && "${CCCL_CI_COMMAND_TIMEOUT}" != "0" ]]; then
            if command -v timeout &> /dev/null; then
                run_command "${group_name}" timeout "${CCCL_CI_COMMAND_TIMEOUT}" "${command[@]}"
                return $?
            fi
            echo "Warning: timeout not found; running without CI timeout." >&2
        fi
    fi

    run_command "${group_name}" "${command[@]}"
}

fail_if_no_gpu() {
    if ! nvidia-smi &> /dev/null; then
        echo "Error: No NVIDIA GPU detected. Please ensure you have an NVIDIA GPU installed and the drivers are properly configured." >&2
        exit 1
    fi
}

function print_test_time_summary()
{
    ctest_log=${1}

    if [ -f ${ctest_log} ]; then
        begin_group "⏱️ Longest Test Steps"
        # Only print the full output in CI:
        if [ -n "${GITHUB_ACTIONS:-}" ]; then
            cmake -DLOGFILE=${ctest_log} -P ../cmake/PrintCTestRunTimes.cmake
        else
            # `|| :` to avoid `set -o pipefail` from triggering when `head` closes the pipe before `cmake` finishes.
            # Otherwise the script will exit early with status 141 (SIGPIPE).
            cmake -DLOGFILE=${ctest_log} -P ../cmake/PrintCTestRunTimes.cmake | head -n 15 || :
        fi
        end_group "⏱️ Longest Test Steps"
    fi
}

function configure_preset()
{
    local BUILD_NAME=$1
    local PRESET=$2
    shift 2
    local CMAKE_OPTIONS=$@

    local GROUP_NAME="🛠️  CMake Configure ${BUILD_NAME}"

    symlink_latest_preset "$PRESET"

    pushd .. > /dev/null
    if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
      # Retry 5 times with 30 seconds between attempts to try to WAR network issues during CPM fetch on CI runners:
      export RUN_COMMAND_RETRY_PARAMS="5 30"
    fi
    status=0
    run_command "$GROUP_NAME" cmake --preset=$PRESET --log-level=VERBOSE $CMAKE_OPTIONS "${GLOBAL_CMAKE_OPTIONS[@]}" || status=$?
    if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
        unset RUN_COMMAND_RETRY_PARAMS
    fi
    popd > /dev/null

    if $CONFIGURE_ONLY; then
        echo "${BUILD_NAME} configuration complete:"
        echo "  Exit code:       ${status}"
        echo "  CMake Preset:    ${PRESET}"
        echo "  CMake Options:   ${CMAKE_OPTIONS}"
        echo "  Build Directory: ${BUILD_DIR}/${PRESET}"
        exit $status
    fi

    return $status
}

function build_preset() {
    local BUILD_NAME=$1
    local PRESET=$2
    local green="1;32"
    local red="1;31"
    local GROUP_NAME="🏗️  Build ${BUILD_NAME}"

    symlink_latest_preset "$PRESET"

    if $CONFIGURE_ONLY; then
        return 0
    fi

    local preset_dir="${BUILD_DIR}/${PRESET}"
    local sccache_json="${preset_dir}/sccache_stats.json"
    local memmon_log="${preset_dir}/memmon.log"

    sccache -z > /dev/null || :

    # Track memory usage on CI:
    if [[ -n "${GITHUB_ACTIONS:-}" || -n "${MEMMON:-}" ]]; then
      util/memmon.sh --start \
          --log-threshold ${MEMMON_LOG_THRESHOLD:-2} \
          --print-threshold ${MEMMON_PRINT_THRESHOLD:-5} \
          --log-file "$memmon_log" \
          --poll ${MEMMON_POLL_INTERVAL:-5} \
          || :
    fi

    pushd .. > /dev/null
    status=0
    run_ci_timed_command "$GROUP_NAME" cmake --build --preset="$PRESET" -v || status=$?
    popd > /dev/null

    if [[ -n "${GITHUB_ACTIONS:-}" || -n "${MEMMON:-}" ]]; then
      util/memmon.sh --stop || :
      run_command "📝 Memory Monitor Log" head -n20 "$memmon_log" || :
    fi

    # Only print detailed stats in actions workflow
    if [ -n "${GITHUB_ACTIONS:-}" ]; then
        sccache --show-adv-stats --stats-format=json > "${sccache_json}" || :
        run_command "📊 sccache stats" sccache --show-adv-stats || :

        begin_group "🥷 ninja build times"
        echo "The "weighted" time is the elapsed time of each build step divided by the number
              of tasks that were running in parallel. This makes it an excellent approximation
              of how "important" a slow step was. A link that is entirely or mostly serialized
              will have a weighted time that is the same or similar to its elapsed time. A
              compile that runs in parallel with 999 other compiles will have a weighted time
              that is tiny."
        ./ninja_summary.py -C ${BUILD_DIR}/${PRESET} || echo "Warning: ninja_summary.py failed to execute properly."
        end_group
    else
      sccache -s || :
    fi

    return $status
}

function test_preset()
{
    local BUILD_NAME=$1
    local PRESET=$2
    local GPU_REQUIRED=${3:-true}

    symlink_latest_preset "$PRESET"

    if $CONFIGURE_ONLY; then
        return 0
    fi

    if $GPU_REQUIRED; then
        fail_if_no_gpu
    fi

    local GROUP_NAME="🚀  Test ${BUILD_NAME}"

    local preset_dir="${BUILD_DIR}/${PRESET}"
    local ctest_log="${preset_dir}/ctest.log"

    pushd .. > /dev/null
    status=0
    run_ci_timed_command "$GROUP_NAME" ctest --output-log "${ctest_log}" --preset="$PRESET" || status=$?
    popd > /dev/null

    print_test_time_summary ${ctest_log}

    return $status
}

function configure_and_build_preset()
{
    local BUILD_NAME=$1
    local PRESET=$2
    shift 2
    local CMAKE_OPTIONS=$@

    configure_preset "$BUILD_NAME" "$PRESET" "$CMAKE_OPTIONS"

    if ! $CONFIGURE_ONLY; then
        build_preset "$BUILD_NAME" "$PRESET"
    fi
}
