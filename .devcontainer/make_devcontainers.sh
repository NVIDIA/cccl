#!/bin/bash

# This script parses the CI matrix.yaml file and generates a devcontainer.json file for each unique combination of
# CUDA version, compiler name/version, and Ubuntu version. The devcontainer.json files are written to the
# .devcontainer directory to a subdirectory named after the CUDA version and compiler name/version.
# GitHub docs on using multiple devcontainer.json files:
# https://docs.github.com/en/codespaces/setting-up-your-project-for-codespaces/adding-a-dev-container-configuration/introduction-to-dev-containers#devcontainerjson

set -euo pipefail

# Ensure the script is being executed in its containing directory
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )";


function usage {
    echo "Usage: $0 [--clean] [-h/--help] [-v/--verbose]"
    echo "  --clean   Remove stale devcontainer subdirectories"
    echo "  -h, --help   Display this help message"
    echo "  -v, --verbose  Enable verbose mode (set -x)"
    exit 1
}

# Function to update the devcontainer.json file with the provided parameters
update_devcontainer() {
    local input_file="$1"
    local output_file="$2"
    local name="$3"
    local cuda_version="$4"
    local compiler_name="$5"
    local compiler_exe="$6"
    local compiler_version="$7"
    local os="$8"
    local devcontainer_version="$9"

    local IMAGE_ROOT="rapidsai/devcontainers:${devcontainer_version}-cpp-"
    local image="${IMAGE_ROOT}${compiler_name}${compiler_version}-cuda${cuda_version}-${os}"

    jq --arg image "$image" --arg name "$name" \
       --arg cuda_version "$cuda_version" --arg compiler_name "$compiler_name" \
       --arg compiler_exe "$compiler_exe" --arg compiler_version "$compiler_version" --arg os "$os" \
       '.image = $image | .name = $name | .containerEnv.DEVCONTAINER_NAME = $name |
        .containerEnv.CCCL_BUILD_INFIX = $name |
        .containerEnv.CCCL_CUDA_VERSION = $cuda_version | .containerEnv.CCCL_HOST_COMPILER = $compiler_name |
        .containerEnv.CCCL_HOST_COMPILER_VERSION = $compiler_version '\
       "$input_file" > "$output_file"
}

make_name() {
    local cuda_version="$1"
    local compiler_name="$2"
    local compiler_version="$3"

    echo "cuda$cuda_version-$compiler_name$compiler_version"
}

CLEAN=false
VERBOSE=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --clean)
            CLEAN=true
            ;;
        -h|--help)
            usage
            ;;
        -v|--verbose)
            VERBOSE=true
            ;;
        *)
            usage
            ;;
    esac
    shift
done

MATRIX_FILE="../ci/matrix.yaml"

# Enable verbose mode if requested
if [ "$VERBOSE" = true ]; then
    set -x
    cat ${MATRIX_FILE}
fi

# Read matrix.yaml and convert it to json
matrix_json=$(yq -o json ${MATRIX_FILE})

# Exclude Windows environments
readonly matrix_json=$(echo "$matrix_json" | jq 'del(.pull_request.nvcc[] | select(.os | contains("windows")))')

# Get the devcontainer image version and define image tag root
readonly DEVCONTAINER_VERSION=$(echo "$matrix_json" | jq -r '.devcontainer_version')

# Get unique combinations of cuda version, compiler name/version, and Ubuntu version
readonly combinations=$(echo "$matrix_json" | jq -c '[.pull_request.nvcc[] | {cuda: .cuda, compiler_name: .compiler.name, compiler_exe: .compiler.exe, compiler_version: .compiler.version, os: .os}] | unique | .[]')

# Update the base devcontainer with the default values
# The root devcontainer.json file is used as the default container as well as a template for all
# other devcontainer.json files by replacing the `image:` field with the appropriate image name
readonly base_devcontainer_file="./devcontainer.json"
readonly NEWEST_GCC_CUDA_ENTRY=$(echo "$combinations" | jq -rs '[.[] | select(.compiler_name == "gcc")] | sort_by((.cuda | tonumber), (.compiler_version | tonumber)) | .[-1]')
readonly DEFAULT_CUDA=$(echo "$NEWEST_GCC_CUDA_ENTRY" | jq -r '.cuda')
readonly DEFAULT_COMPILER_NAME=$(echo "$NEWEST_GCC_CUDA_ENTRY" | jq -r '.compiler_name')
readonly DEFAULT_COMPILER_EXE=$(echo "$NEWEST_GCC_CUDA_ENTRY" | jq -r '.compiler_exe')
readonly DEFAULT_COMPILER_VERSION=$(echo "$NEWEST_GCC_CUDA_ENTRY" | jq -r '.compiler_version')
readonly DEFAULT_OS=$(echo "$NEWEST_GCC_CUDA_ENTRY" | jq -r '.os')
readonly DEFAULT_NAME=$(make_name "$DEFAULT_CUDA" "$DEFAULT_COMPILER_NAME" "$DEFAULT_COMPILER_VERSION")

update_devcontainer ${base_devcontainer_file} "./temp_devcontainer.json" "$DEFAULT_NAME" "$DEFAULT_CUDA" "$DEFAULT_COMPILER_NAME" "$DEFAULT_COMPILER_EXE" "$DEFAULT_COMPILER_VERSION" "$DEFAULT_OS" "$DEVCONTAINER_VERSION"
mv "./temp_devcontainer.json" ${base_devcontainer_file}

# Create an array to keep track of valid subdirectory names
valid_subdirs=()

# The img folder should not be removed:
valid_subdirs+=("img")

# For each unique combination
for combination in $combinations; do
    cuda_version=$(echo "$combination" | jq -r '.cuda')
    compiler_name=$(echo "$combination" | jq -r '.compiler_name')
    compiler_exe=$(echo "$combination" | jq -r '.compiler_exe')
    compiler_version=$(echo "$combination" | jq -r '.compiler_version')
    os=$(echo "$combination" | jq -r '.os')

    name=$(make_name "$cuda_version" "$compiler_name" "$compiler_version")
    mkdir -p "$name"
    new_devcontainer_file="$name/devcontainer.json"

    update_devcontainer "$base_devcontainer_file" "$new_devcontainer_file" "$name" "$cuda_version" "$compiler_name" "$compiler_exe" "$compiler_version" "$os" "$DEVCONTAINER_VERSION"
    echo "Created $new_devcontainer_file"

    # Add the subdirectory name to the valid_subdirs array
    valid_subdirs+=("$name")
done

# Clean up stale subdirectories and devcontainer.json files
if [ "$CLEAN" = true ]; then
    for subdir in ./*; do
        if [ -d "$subdir" ] && [[ ! " ${valid_subdirs[@]} " =~ " ${subdir#./} " ]]; then
            echo "Removing stale subdirectory: $subdir"
            rm -r "$subdir"
        fi
    done
fi
