#!/bin/bash

source "$(dirname "$0")/build_common.sh"

print_environment_details

PRESET="libcudacxx-cpp${CXX_STANDARD}"
CMAKE_OPTIONS=""

configure_preset libcudacxx "$PRESET" "$CMAKE_OPTIONS"

# The libcudacxx tests are split into two presets, one for
# regular ctest tests and another that invokes the lit tests
# harness with extra options for verbosity, etc:
CTEST_PRESET="libcudacxx-ctest-cpp${CXX_STANDARD}"
LIT_PRESET="libcudacxx-lit-cpp${CXX_STANDARD}"

test_preset "libcudacxx (CTest)" ${CTEST_PRESET}

source "./sccache_stats.sh" "start"
test_preset "libcudacxx (lit)" ${LIT_PRESET}
source "./sccache_stats.sh" "end"

print_time_summary
