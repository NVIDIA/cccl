#!/bin/bash

base_devcontainer_file="devcontainer_template.json"

# Define image root
IMAGE_ROOT="rapidsai/devcontainers:23.06-cpp-"

# Read matrix.yaml and convert it to json
matrix_json=$(yq -o json ../ci/matrix.yaml)

# Get unique combinations of cuda version, compiler name/version, and Ubuntu version
combinations=$(echo "$matrix_json" | jq -c '[."pull-request"[] | {cuda: .cuda, compiler_name: .compiler.name, compiler_version: .compiler.version, ubuntu: .ubuntu}] | unique | .[]')

# For each unique combination
for combination in $combinations; do
    cuda_version=$(echo "$combination" | jq -r '.cuda')
    compiler_name=$(echo "$combination" | jq -r '.compiler_name')
    compiler_version=$(echo "$combination" | jq -r '.compiler_version')
    ubuntu_version=$(echo "$combination" | jq -r '.ubuntu')

    directory="cuda$cuda_version-$compiler_name$compiler_version"

    mkdir -p "$directory"

    devcontainer_file="$directory/devcontainer.json"

    image_name="$IMAGE_ROOT$compiler_name$compiler_version-cuda$cuda_version-ubuntu$ubuntu_version"

    # Use the base_devcontainer.json as a template, plug in the CUDA, compiler names, versions, and Ubuntu version,
    # and write the output to the new devcontainer.json file
    jq --arg image_name "$image_name" '. + {image: $image_name}' $base_devcontainer_file > "$devcontainer_file"

    echo "Created $devcontainer_file"
done
