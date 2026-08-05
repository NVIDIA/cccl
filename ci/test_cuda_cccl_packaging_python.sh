#!/usr/bin/env bash

set -euo pipefail

ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$ci_dir/.." && pwd)"

# shellcheck source=ci/pyenv_helper.sh
source "$ci_dir/pyenv_helper.sh"
# shellcheck source=ci/util/python/common_arg_parser.sh
source "$ci_dir/util/python/common_arg_parser.sh"
parse_python_args "$@"
require_py_version "Usage: $0 -py-version <python_version>"
setup_python_env "${py_version}"

python -m pip install pytest 'tomli; python_version < "3.11"'
python -m pytest -q \
  "${repo_root}/python/cuda_cccl/tests/packaging" \
  "${repo_root}/python/cuda_cccl_meta/tests"
