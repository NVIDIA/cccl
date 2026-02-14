#!/usr/bin/env bash

# CI wrapper for the `bisect` project test job.
# Invokes ci/util/git_bisect.sh with the provided arguments to
# run a bisection on a GPU runner.

set -euo pipefail

ci_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd "${ci_dir}/.." && pwd)

user_args=("$@")
set --
source "${ci_dir}/build_common.sh"
set -- "${user_args[@]}"

cd "${repo_dir}"
cmd=("${ci_dir}/util/git_bisect.sh" "$@")
printf '\033[34m%s\033[0m\n' "${cmd[*]}"
"${cmd[@]}"
