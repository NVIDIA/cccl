#!/bin/bash

function usage {
    echo "Usage: $0"
    echo
    echo "This script is intended to be run within one of CCCL's Dev Containers."
    echo "It verifies that the expected environment variables and binary versions match what is expected."
}

check_envvars() {
    for var_name in "$@"; do
        if [[ -z "${!var_name:-}" ]]; then
            echo "::error:: ${var_name} variable is not set."
            exit 1
        else
            echo "$var_name=${!var_name}"
        fi
    done
}

check_host_compiler_version() {
    local version_output=$($CXX --version)

    if [[ "$CXX" == "g++" ]]; then
        local actual_version=$(echo "$version_output" | head -n 1 | cut -d ' ' -f 4 | cut -d '.' -f 1)
        local expected_compiler="gcc"
    elif [[ "$CXX" == "clang++" ]]; then
        if [[ $version_output =~ clang\ version\ ([0-9]+) ]]; then
            actual_version=${BASH_REMATCH[1]}
        else
            echo "::error:: Unable to determine clang version."
            exit 1
        fi
        expected_compiler="llvm"
    elif [[ "$CXX" == "icpc" ]]; then
        local actual_version=$(echo "$version_output" | head -n 1 | cut -d ' ' -f 3 )
        # The icpc compiler version of oneAPI release 2023.2.0 is 2021.10.0
        if [[ "$actual_version" == "2021.10.0" ]]; then
            actual_version="2023.2.0"
        fi
        expected_compiler="oneapi"
    else
        echo "::error:: Unexpected CXX value ($CXX)."
        exit 1
    fi

    if [[ "$expected_compiler" != "${CCCL_HOST_COMPILER}" || "$actual_version" != "$CCCL_HOST_COMPILER_VERSION" ]]; then
        echo "::error:: CXX ($CXX) version ($actual_version) does not match the expected compiler (${CCCL_HOST_COMPILER}) and version (${CCCL_HOST_COMPILER_VERSION})."
        exit 1
    else
        echo "Detected host compiler: $CXX version $actual_version"
    fi
}

check_cuda_version() {
    local cuda_version_output=$(nvcc --version)
    if [[ $cuda_version_output =~ release\ ([0-9]+\.[0-9]+) ]]; then
        local actual_cuda_version=${BASH_REMATCH[1]}
    else
        echo "::error:: Unable to determine CUDA version from nvcc."
        exit 1
    fi

    if [[ "$actual_cuda_version" != "$CCCL_CUDA_VERSION" ]]; then
        echo "::error:: CUDA version ($actual_cuda_version) does not match the expected CUDA version ($CCCL_CUDA_VERSION)."
        exit 1
    else
        echo "Detected CUDA version: $actual_cuda_version"
    fi
}

main() {
    if [[ "$1" == "-h" || "$1" == "--help" ]]; then
        usage
        exit 0
    fi

    set -euo pipefail

    check_envvars DEVCONTAINER_NAME CXX CUDAHOSTCXX CCCL_BUILD_INFIX CCCL_HOST_COMPILER CCCL_CUDA_VERSION CCCL_HOST_COMPILER_VERSION

    check_host_compiler_version

    check_cuda_version

    echo "Dev Container successfully verified!"
}

main "$@"
