#!/usr/bin/env bash
# Experiment-only execution of the lit stage after the separate CTest stage.
set -euo pipefail

ci_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd "${ci_dir}/.." && pwd)

# shellcheck source=ci/build_common.sh
source "${ci_dir}/build_common.sh"
print_environment_details

producer_id="$(util/workflow/get_producer_id.sh)"
artifact="z_libcudacxx-test-artifacts-${DEVCONTAINER_NAME:?}-$producer_id"
run_command "Unpacking artifact '$artifact'" \
  "${ci_dir}/util/artifacts/download_packed.sh" "$artifact" "${repo_dir}"

export LIT_OPTS="${LIT_OPTS:+${LIT_OPTS} }-Dtest_executable_mode=replay"
test_preset "libcudacxx (lit replay)" "libcudacxx-lit"
print_time_summary
