#!/bin/bash

set -euo pipefail

source "$(dirname "$0")/build_common.sh"

print_environment_details

fail_if_no_gpu

source "test_python_common.sh"

list_environment

run_tests "cuda_cccl"
