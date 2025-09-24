#!/usr/bin/env bash

# CI wrapper for the `bisect` project build job.
# Forwards all arguments to ci/util/git_bisect.sh to run a bisection
# using the provided configuration/build/test options.

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
