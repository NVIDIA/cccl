#!/usr/bin/env bash

# CI wrapper for the `target` project test job.
# Invokes ci/util/build_and_test_targets.sh with the provided arguments to
# build and test selected targets on a GPU runner.

set -euo pipefail

ci_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd "${ci_dir}/.." && pwd)

user_args=("$@")
set --
source "${ci_dir}/build_common.sh"
set -- "${user_args[@]}"

cd "${repo_dir}"
cmd=("${ci_dir}/util/build_and_test_targets.sh" "$@")
printf '\033[34m%s\033[0m\n' "${cmd[*]}"
"${cmd[@]}"
