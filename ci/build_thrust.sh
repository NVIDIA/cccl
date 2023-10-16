#!/bin/bash

source "$(dirname "$0")/build_common.sh"

PRESET="thrust-cpp$CXX_STANDARD"

CMAKE_OPTIONS=""

configure_and_build_preset "Thrust" "$PRESET" "$CMAKE_OPTIONS"
