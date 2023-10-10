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
    else
        echo "::error:: Unexpected CXX value ($CXX)."
        exit 1
    fi

    if [[ "$expected_compiler" != "${CCCL_HOST_COMPILER}" || "$actual_version" != "$CCCL_HOST_COMPILER_VERSION" ]]; then
        echo "::error:: CXX ($CXX) version ($actual_version) does not match the expected compiler (${CCCL_HOST_COMPILER}) and version (${CCCL_HOST_COMPILER_VERSION})."
        exit 1          set -euo pipefail
          if [[ -z "${CXX:-}" ]]; then
            echo "::error:: CXX variable is not set."
            exit 1
          fi
          $CXX --version
          version_output=$($CXX --version)
          if [[ "$CXX" == "g++" ]]; then
              # Extracts major version for g++ output format
              actual_version=$(echo "$version_output" | head -n 1 | cut -d ' ' -f 4 | cut -d '.' -f 1)
              expected_compiler="gcc"
          elif [[ "$CXX" == "clang++" ]]; then
              # Extracts major version for clang++ output format
              if [[ $version_output =~ clang\ version\ ([0-9]+) ]]; then
                  actual_version=${BASH_REMATCH[1]}
              else
                  echo "::error:: Unable to determine clang version."
                  exit 1
              fi
              expected_compiler="llvm"
          else
              echo "::error:: Unexpected CXX value ($CXX)."
              exit 1
          fi

          expected_version="${CCCL_HOST_COMPILER_VERSION}"

          if [[ "$expected_compiler" != "${CCCL_HOST_COMPILER}" || "$actual_version" != "$expected_version" ]]; then
              echo "::error:: CXX ($CXX) version ($actual_version) does not match the expected compiler (${CCCL_HOST_COMPILER}) and version (${CCCL_HOST_COMPILER_VERSION})."
              exit 1
          fi

          # Check CUDA version from nvcc
          cuda_version_output=$(nvcc --version)
          # Extract the MAJOR.MINOR version from the output
          if [[ $cuda_version_output =~ release\ ([0-9]+\.[0-9]+) ]]; then
              actual_cuda_version=${BASH_REMATCH[1]}
          else
              echo "::error:: Unable to determine CUDA version from nvcc."
              exit 1
          fi

          expected_cuda_version="${CCCL_CUDA_VERSION}"
          if [[ "$actual_cuda_version" != "$expected_cuda_version" ]]; then
              echo "::error:: CUDA version ($actual_cuda_version) does not match the expected CUDA version ($expected_cuda_version)."
              exit 1
          fi
ion=${BASH_REMATCH[1]}
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

    check_envvars DEVCONTAINER_NAME CXX CCCL_HOST_COMPILER CCCL_CUDA_VERSION CCCL_HOST_COMPILER_VERSION

    check_host_compiler_version

    check_cuda_version

    echo "Dev Container successfully verified!"
}

main "$@"
