#!/bin/bash

source "$(dirname "$0")/build_common.sh"

print_environment_details

# CUB benchmarks require at least CUDA nvcc 11.5 for int128
# Returns "true" if the first version is greater than or equal to the second
version_compare() {
    if [[ "$(printf '%s\n' "$1" "$2" | sort -V | head -n1)" == "$2" ]]; then
        echo "true"
    else
        echo "false"
    fi
}

ENABLE_CUB_BENCHMARKS="false"
ENABLE_CUB_RDC="false"

if [[ "$CUDA_COMPILER" == *nvcc* ]]; then
    ENABLE_CUB_RDC="true"
    NVCC_VERSION=$($CUDA_COMPILER --version | grep release | awk '{print $6}' | cut -c2-)
    if [[ -n "${DISABLE_CUB_BENCHMARKS}" ]]; then
        echo "Benchmarks have been forcefully disabled."
    elif [[ $(version_compare $NVCC_VERSION 11.5) == "true" ]]; then
        ENABLE_CUB_BENCHMARKS="true"
        echo "nvcc version is $NVCC_VERSION. Building CUB benchmarks."
    else
        echo "nvcc version is $NVCC_VERSION. Not building CUB benchmarks because nvcc version is less than 11.5."
    fi
else
    echo "Not building with NVCC, disabling RDC and benchmarks."
fi

if [[ "$HOST_COMPILER" == *icpc* ]]; then
    ENABLE_CUB_BENCHMARKS="false"
fi

PRESET="cub-cpp$CXX_STANDARD"

CMAKE_OPTIONS="
    -DCUB_ENABLE_BENCHMARKS="$ENABLE_CUB_BENCHMARKS"\
    -DCUB_ENABLE_RDC_TESTS="$ENABLE_CUB_RDC" \
"

configure_and_build_preset "CUB" "$PRESET" "$CMAKE_OPTIONS"

create_test_artifact() {


    # TODO REMOVE THIS
    # Just testing for CI, we need to add this to the image.
    if ! command -v pbzip2 &> /dev/null; then
        sudo apt update
        sudo apt-get install -y pbzip2
    fi


    local start_time=$SECONDS
    local preset_dir="${BUILD_DIR}/${PRESET}"
    local temp_dir="./output_artifact"
    local artifact_name="output_artifact.tar.bz2"

    pushd ".." > /dev/null

    # Make preset_dir relative to cwd
    preset_dir="$(realpath --relative-to=. "$preset_dir")"

    echo "Current directory: $(pwd)"
    echo "Preset directory: $preset_dir"
    echo "Temp directory: $temp_dir"
    echo "Artifact name: $artifact_name"
    mkdir -p "$temp_dir"

    # Stage files that match regexes
    for regex in "$@"; do
        regex="^\./$preset_dir/$regex"
        echo "Regex: $regex"
        find . -type f -regex "$regex" -exec cp --parents {} "$temp_dir" \;
    done

    # Parallelize compression if pbzip2 is not available
    if command -v pbzip2 &> /dev/null; then
        # Benchmark on 64 core: 17s
        tar -cv -C "$temp_dir" . | pbzip2 -c > output_artifact.tar.bz2
    else
        # Benchmark on 64 core: 4m32s
        tar -cvjf output_artifact.tar.bz2 -C "$temp_dir" .
    fi

    # Clean up the temporary directory
    rm -rf "$temp_dir"

    echo "Testing artifact created as $(pwd)/output_artifact.tar.bz2."
    echo "Size: $(du -h output_artifact.tar.bz2 | awk '{print $1}')"
    md5sum output_artifact.tar.bz2

    popd > /dev/null

    local end_time=$SECONDS
    echo "Time taken to create testing artifact: $((end_time - start_time))s"
}

# The preset dir will be prepended to the regexes:
create_test_artifact \
    'build\.ninja$' \
    '.*rules\.ninja$' \
    'CMakeCache\.txt$' \
    '.*VerifyGlobs\.cmake$' \
    '.*CTestTestfile\.cmake$' \
    'bin/.*' \
    'lib/.*'

print_time_summary
