#!/usr/bin/env bash

cd "$(dirname "${BASH_SOURCE[0]}")"
# shellcheck source=ci/build_common.sh
source "./build_common.sh"

print_environment_details

"./build_libcudacxx.sh" "$@" "-cmake-options" "-DLIBCUDACXX_SKIP_LIT_BUILD=1"

# test_preset "libcudacxx (CTest)" "libcudacxx-ctest"

# Reset sccache stats to get lit build+test time
sccache -z > /dev/null || :
test_preset "libcudacxx (lit)" "libcudacxx-lit"
sccache --show-adv-stats || :

print_time_summary
