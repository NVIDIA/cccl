#!/bin/bash

set -euo pipefail

# Ensure the script is being executed in the root cccl directory:
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/..";

# Get the current CCCL info:
readonly cccl_repo="${PWD}"
readonly workdir="${cccl_repo}/test/stdpar"

CXX_STANDARD=17

args=("$@")
while [ "${#args[@]}" -ne 0 ]; do
    case "${args[0]}" in
    -std)  CXX_STANDARD="${args[1]}";  args=("${args[@]:2}");;
    *) echo "Unrecognized option: ${args[0]}"; exit 1 ;;
    esac
done

mkdir -p "${workdir}"
cd "${workdir}"

# Configure and build
rm -rf build

cmake -B build -S . -G Ninja \
  -DCMAKE_CXX_STANDARD="${CXX_STANDARD}" \
  `# Explicitly compile for hopper since the CI machine does not have a gpu:` \
  -DCMAKE_CXX_FLAGS="-gpu=cc90"

cmake --build build -j ${PARALLEL_LEVEL:-}
