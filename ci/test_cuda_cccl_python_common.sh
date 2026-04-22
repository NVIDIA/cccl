#!/bin/bash

set -euo pipefail

function install_cuda_cccl_wheel_for_tests() {
  local wheel_path="$1"
  local cuda_major_version="$2"

  python -m pip install "${wheel_path}[test-cu${cuda_major_version}]"
}
