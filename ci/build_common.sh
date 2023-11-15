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

# begin_group: Start a named section of log output, possibly with color.
# Usage: begin_group "Group Name" [Color]
#   Group Name: A string specifying the name of the group.
#   Color (optional): ANSI color code to set text color. Default is blue (1;34).
function begin_group() {
    # See options for colors here: https://gist.github.com/JBlond/2fea43a3049b38287e5e9cefc87b2124
    local blue="34"
    local name="${1:-}"
    local color="${2:-$blue}"

    if [ -n "${GITHUB_ACTIONS:-}" ]; then
        echo -e "::group::\e[${color}m${name}\e[0m"
    else
        echo -e "\e[${color}m================== ${name} ======================\e[0m"
    fi
}

# end_group: End a named section of log output and print status based on exit status.
# Usage: end_group "Group Name" [Exit Status]
#   Group Name: A string specifying the name of the group.
#   Exit Status (optional): The exit status of the command run within the group. Default is 0.
function end_group() {
    local name="${1:-}"
    local build_status="${2:-0}"
    local duration="${3:-}"
    local red="31"
    local blue="34"

    if [ -n "${GITHUB_ACTIONS:-}" ]; then
        echo "::endgroup::"

        if [ "$build_status" -ne 0 ]; then
            echo -e "::error::\e[${red}m ${name} - Failed (⬆️ click above for full log ⬆️)\e[0m"
        fi
    else
        if [ "$build_status" -ne 0 ]; then
            echo -e "\e[${red}m================== End ${name} - Failed${duration:+ - Duration: ${duration}s} ==================\e[0m"
        else
            echo -e "\e[${blue}m================== End ${name} - Success${duration:+ - Duration: ${duration}s} ==================\n\e[0m"
        fi
    fi
}

declare -A command_durations

# Runs a command within a named group, handles the exit status, and prints appropriate messages based on the result.
# Usage: run_command "Group Name" command [arguments...]
function run_command() {
    local group_name="${1:-}"
    shift
    local command=("$@")
    local status

    begin_group "$group_name"
    set +e
    local start_time=$(date +%s)
    "${command[@]}"
    status=$?
    local end_time=$(date +%s)
    set -e
    local duration=$((end_time - start_time))
    end_group "$group_name" $status $duration
    command_durations["$group_name"]=$duration
    return $status
}

function string_width() {
    local str="$1"
    echo "$str" | awk '{print length}'
}

function print_time_summary() {
    local max_length=0
    local group

    # Find the longest group name for formatting
    for group in "${!command_durations[@]}"; do
        local group_length=$(echo "$group" | awk '{print length}')
        if [ "$group_length" -gt "$max_length" ]; then
            max_length=$group_length
        fi
    done

    echo "Time Summary:"
    for group in "${!command_durations[@]}"; do
        printf "%-${max_length}s : %s seconds\n" "$group" "${command_durations[$group]}"
    done

    # Clear the array of timing info
    declare -gA command_durations=()
}


print_environment_details() {
  begin_group "⚙️ Environment Details"

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

  echo "Current commit is:"
  git log -1 || echo "Not a repository"

  if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
  else
    echo "nvidia-smi not found"
  fi

  end_group "⚙️ Environment Details"
}


function configure_preset()
{
    local BUILD_NAME=$1
    local PRESET=$2
    local CMAKE_OPTIONS=$3
    local GROUP_NAME="🛠️  CMake Configure ${BUILD_NAME}"

    pushd .. > /dev/null
    run_command "$GROUP_NAME" cmake --preset=$PRESET --log-level=VERBOSE $GLOBAL_CMAKE_OPTIONS $CMAKE_OPTIONS
    status=$?
    popd > /dev/null
    return $status
}

function build_preset() {
    local BUILD_NAME=$1
    local PRESET=$2
    local green="1;32"
    local red="1;31"
    local GROUP_NAME="🏗️  Build ${BUILD_NAME}"

    source "./sccache_stats.sh" "start"

    pushd .. > /dev/null
    run_command "$GROUP_NAME" cmake --build --preset=$PRESET -v
    status=$?
    popd > /dev/null

    minimal_sccache_stats=$(source "./sccache_stats.sh" "end")

    # Only print detailed stats in actions workflow
    if [ -n "${GITHUB_ACTIONS:-}" ]; then
        begin_group "💲 sccache stats"
        echo "${minimal_sccache_stats}"
        sccache -s
        end_group

        begin_group "🥷 ninja build times"
        echo "The "weighted" time is the elapsed time of each build step divided by the number
              of tasks that were running in parallel. This makes it an excellent approximation
              of how "important" a slow step was. A link that is entirely or mostly serialized
              will have a weighted time that is the same or similar to its elapsed time. A
              compile that runs in parallel with 999 other compiles will have a weighted time
              that is tiny."
        ./ninja_summary.py -C ${BUILD_DIR}/${PRESET}
        end_group
    else
      echo $minimal_sccache_stats
    fi

    return $status
}

function test_preset()
{
    local BUILD_NAME=$1
    local PRESET=$2
    local GROUP_NAME="🚀  Test ${BUILD_NAME}"

    pushd .. > /dev/null
    run_command "$GROUP_NAME" ctest --preset=$PRESET
    status=$?
    popd > /dev/null
    return $status
}

function configure_and_build_preset()
{
    local BUILD_NAME=$1
    local PRESET=$2
    local CMAKE_OPTIONS=$3

    configure_preset "$BUILD_NAME" "$PRESET" "$CMAKE_OPTIONS"
    build_preset "$BUILD_NAME" "$PRESET"
}
