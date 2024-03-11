#!/bin/bash
set -eo pipefail

# Ensure the script is being executed in its containing directory
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )";

# TBD: verbose? any extra args?

source ./pretty_printing.sh

pushd .. > /dev/null
GROUP_NAME="🛠️  CMake Configure Libcudacxx Codegen"
run_command "$GROUP_NAME" cmake --preset libcudacxx-codegen
status=$?
popd > /dev/null

pushd .. > /dev/null
GROUP_NAME="🏗️  Build Libcudacxx Codegen"
run_command "$GROUP_NAME" cmake --build --preset libcudacxx-codegen

status=$?
popd > /dev/null

pushd .. > /dev/null
GROUP_NAME="🚀  Test Libcudacxx Codegen"
run_command "$GROUP_NAME" ctest --preset libcudacxx-codegen
status=$?
popd > /dev/null
