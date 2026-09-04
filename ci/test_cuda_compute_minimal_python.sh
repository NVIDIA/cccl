#!/usr/bin/env bash

set -euo pipefail

ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ci_dir/pyenv_helper.sh"
# shellcheck source=ci/util/python/common_arg_parser.sh
source "$ci_dir/util/python/common_arg_parser.sh"
parse_python_args "$@"

require_py_version "Usage: $0 -py-version <python_version>"

dispatch_python_lane "ci/util/python/run_compute_minimal_tests.sh" "$@"
