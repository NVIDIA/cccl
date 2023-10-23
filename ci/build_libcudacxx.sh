#!/bin/bash

source "$(dirname "$0")/build_common.sh"

PRESET="libcudacxx-cpp${CXX_STANDARD}"
CMAKE_OPTIONS=""

configure_and_build_preset libcudacxx "$PRESET" "$CMAKE_OPTIONS"
