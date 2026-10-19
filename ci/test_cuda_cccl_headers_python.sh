#!/usr/bin/env bash

set -euo pipefail

ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ci_dir/pyenv_helper.sh"
# shellcheck source=ci/util/python/common_arg_parser.sh
source "$ci_dir/util/python/common_arg_parser.sh"
parse_python_args "$@"

# The only lane on this path that does not need a GPU: it asserts that headers
# shipped in the wheel are on disk, and never launches a kernel.
export CCCL_MINIMAL_CONTAINER_NO_GPU=1

dispatch_python_lane "ci/util/python/run_headers_tests.sh" "$@"
