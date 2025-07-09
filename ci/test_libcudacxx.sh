#!/bin/bash

cd "$(dirname "${BASH_SOURCE[0]}")"
source "./build_common.sh"

print_environment_details

"./build_libcudacxx.sh" "$@"

PRESET="libcudacxx-cpp${CXX_STANDARD}"
CMAKE_OPTIONS=""

configure_preset libcudacxx "$PRESET" "$CMAKE_OPTIONS"

test_preset "libcudacxx (CTest)" "libcudacxx-ctest-cpp${CXX_STANDARD}"

source "./sccache_stats.sh" "start" || :
test_preset "libcudacxx (lit)" "libcudacxx-lit-cpp${CXX_STANDARD}"
source "./sccache_stats.sh" "end" || :

print_time_summary
