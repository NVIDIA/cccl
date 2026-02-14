#!/bin/bash

cd "$(dirname "${BASH_SOURCE[0]}")"
source "./build_common.sh"

print_environment_details

"./build_libcudacxx.sh" "$@"

PRESET="libcudacxx"
CMAKE_OPTIONS="-DCMAKE_CXX_STANDARD=${CXX_STANDARD} -DCMAKE_CUDA_STANDARD=${CXX_STANDARD}"

configure_preset libcudacxx "$PRESET" "$CMAKE_OPTIONS"

test_preset "libcudacxx (CTest)" "libcudacxx-ctest"

sccache -z > /dev/null || :
test_preset "libcudacxx (lit)" "libcudacxx-lit"
sccache --show-adv-stats || :

print_time_summary
