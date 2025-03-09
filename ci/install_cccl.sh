#!/bin/bash

set -eo pipefail

target_dir=$(realpath "$1")
mkdir -p "$target_dir"

# Move script to the root directory of the project
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )";

source ./pretty_printing.sh

# Check if the correct number of arguments has been provided
function usage {
    echo "Usage: $0 [OPTIONS] dir"
    echo
    echo "Installs CCCL to the provided directory"

    echo "Options:"
    echo "  -v/-verbose: Enable shell echo for debugging"
    echo
    echo "Examples:"
    echo "  $ $0 ~/my/prefix"
    exit 1
}

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --verbose) VERBOSE=true; ;;
        -v) VERBOSE=true; ;;
        *) break ;;
    esac
    shift
done

if [ $VERBOSE ]; then
    set -x
fi

# Move to cccl/ dir
pushd ".." > /dev/null
GROUP_NAME="🛠️  CMake Configure CCCL - Install"
run_command "$GROUP_NAME" cmake -G "Unix Makefiles" --preset install -DCMAKE_INSTALL_PREFIX="${target_dir}"
status=$?

GROUP_NAME="🏗️  Install CCCL"
run_command "$GROUP_NAME" cmake --build --preset install --target install
status=$?
popd > /dev/null
