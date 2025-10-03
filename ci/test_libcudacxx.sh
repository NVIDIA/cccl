#!/bin/bash

cd "$(dirname "${BASH_SOURCE[0]}")"
source "./build_common.sh"

print_environment_details

"./build_libcudacxx.sh" "$@"

PRESET="libcudacxx-cpp${CXX_STANDARD}"
CMAKE_OPTIONS=""

configure_preset libcudacxx "$PRESET" "$CMAKE_OPTIONS"

test_preset "libcudacxx (CTest)" "libcudacxx-ctest-cpp${CXX_STANDARD}"

sccache -z > /dev/null || :
test_preset "libcudacxx (lit)" "libcudacxx-lit-cpp${CXX_STANDARD}"
sccache --show-adv-stats || :

print_time_summary
