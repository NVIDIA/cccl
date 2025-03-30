#!/bin/bash

source "$(dirname "$0")/build_common.sh"

print_environment_details

./build_lib_ptx_json.sh "$@"

PRESET="cccl-lib-ptx-json"

test_preset "CCCL PTX-JSON Library" ${PRESET}

print_time_summary
