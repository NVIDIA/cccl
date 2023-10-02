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
matrix_json=$(echo "$matrix_json" | jq 'del(.pull_request.nvcc[] | select(.os | contains("windows")))')

# Get the devcontainer image version and define image tag root
DEVCONTAINER_VERSION=$(echo "$matrix_json" | jq -r '.devcontainer_version')
IMAGE_ROOT="rapidsai/devcontainers:${DEVCONTAINER_VERSION}-cpp-"

# The root devcontainer.json file is used as a template for all other devcontainer.json files
# by replacing the `image:` field with the appropriate image name
base_devcontainer_file="./devcontainer.json"
# Update the top-level devcontainer.json with the new version
jq --arg version "$DEVCONTAINER_VERSION" '.version = $version' $base_devcontainer_file > tmp_devcontainer.json && mv tmp_devcontainer.json $base_devcontainer_file

# Get unique combinations of cuda version, compiler name/version, and Ubuntu version
combinations=$(echo "$matrix_json" | jq -c '[.pull_request.nvcc[] | {cuda: .cuda, compiler_name: .compiler.name, compiler_version: .compiler.version, os: .os}] | unique | .[]')

# Create an array to keep track of valid subdirectory names
valid_subdirs=()

# For each unique combination
for combination in $combinations; do
    cuda_version=$(echo "$combination" | jq -r '.cuda')
    compiler_name=$(echo "$combination" | jq -r '.compiler_name')
    compiler_version=$(echo "$combination" | jq -r '.compiler_version')
    os=$(echo "$combination" | jq -r '.os')

    name="cuda$cuda_version-$compiler_name$compiler_version"
    mkdir -p "$name"
    devcontainer_file="$name/devcontainer.json"
    image="$IMAGE_ROOT$compiler_name$compiler_version-cuda$cuda_version-$os"

    # Use the base devcontainer.json as a template, plug in the CUDA, compiler names, versions, and Ubuntu version,
    # and write the output to the new devcontainer.json file
    #jq --arg image "$image"  --arg name "$name" '. + {image: $image, name: $name}' $base_devcontainer_file > "$devcontainer_file"
    jq --arg image "$image" --arg name "$name" '.image = $image | .name = $name | .containerEnv.DEVCONTAINER_NAME = $name' $base_devcontainer_file > "$devcontainer_file"

    echo "Created $devcontainer_file"
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
