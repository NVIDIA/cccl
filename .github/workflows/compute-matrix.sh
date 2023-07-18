#!/bin/bash

set -euo pipefail

# Ensure the script is being executed in its containing directory
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )";

# Check for the correct number of arguments
if [ $# -ne 2 ]; then
    echo "Usage: $0 MATRIX_FILE MATRIX_QUERY"
    echo "MATRIX_FILE: The path to the matrix file."
    echo "MATRIX_QUERY: The jq query used to specify the desired matrix. e.g., '.["pull-request"]["nvcc]"'"
    exit 1
fi

MATRIX_FILE="$1"
MATRIX_QUERY="$2"

# Use /dev/null as the default value for GITHUB_OUTPUT if it isn't set, i.e., not running in GitHub Actions
GITHUB_OUTPUT="${GITHUB_OUTPUT:-/dev/null}"

echo "Input matrix file:"
cat "$MATRIX_FILE"
echo "Query: $MATRIX_QUERY"
echo $(yq -o=json "$MATRIX_FILE" | jq -c -r "$MATRIX_QUERY | map(. as \$o | {std: .std[]} + del(\$o.std))")

#FULL_MATRIX=$(yq -o=json "$MATRIX_FILE" | jq -c --arg matrix_type "$MATRIX_QUERY" '[ .[$matrix_type][] | . as $o | {std: .std[]} + del($o.std)]')
#echo "FULL_MATRIX=$FULL_MATRIX" | tee -a "$GITHUB_OUTPUT"
#CUDA_VERSIONS=$(echo $FULL_MATRIX | jq -c '[.[] | .cuda] | unique')
#echo "CUDA_VERSIONS=$CUDA_VERSIONS" | tee -a "$GITHUB_OUTPUT"
#COMPILERS=$(echo $FULL_MATRIX | jq -c '[.[] | .compiler.name] | unique')
#echo "COMPILERS=$COMPILERS" | tee -a "$GITHUB_OUTPUT"
#PER_CUDA_COMPILER_MATRIX=$(echo $FULL_MATRIX | jq -c ' group_by(.cuda + .compiler.name) | map({(.[0].cuda + "-" + .[0].compiler.name): .}) | add')
#echo "PER_CUDA_COMPILER_MATRIX=$PER_CUDA_COMPILER_MATRIX" | tee -a "$GITHUB_OUTPUT"