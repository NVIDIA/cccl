#!/usr/bin/env bash
# Experiment-only isolated replay of the single CUDA allocation failure.
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

executable="${repo_dir}/build/libcudacxx/libcudacxx/test/libcudacxx/test/cuda/memory/Output/is_pointer_accessible.pass.cpp.exe"
status=0
run_command "Isolated executable through Bash" "$executable" || status=$?
run_command "Isolated executable through PowerShell" powershell.exe \
  -NoProfile -ExecutionPolicy Bypass \
  -File "$(cygpath -w "${ci_dir}/run_native_executable_baseline.ps1")" \
  -TestExecutable "$(cygpath -w "$executable")" || status=$?
exit "$status"
