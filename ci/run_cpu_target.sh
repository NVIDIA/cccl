#!/usr/bin/env bash

# CI wrapper for the `target` project build job.
# Forwards all arguments to ci/util/build_and_test_targets.sh to configure
# and build selected targets on a CPU runner.

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
