#!/bin/bash

set -euo pipefail

# Check for the correct number of arguments
if [ $# -ne 2 ]; then
    echo "Usage: $0 MATRIX_FILE MATRIX_QUERY"
    echo "MATRIX_FILE: The path to the matrix file."
    echo "MATRIX_QUERY: The jq query used to specify the desired matrix. e.g., '.pull-request.nvcc'"
    exit 1
fi

# Get realpath before changing directory
MATRIX_FILE=$(realpath "$1")
MATRIX_QUERY="$2"

# Ensure the script is being executed in its containing directory
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )";

echo "Input matrix file:" >&2
cat "$MATRIX_FILE" >&2
echo "Query: $MATRIX_QUERY" >&2
echo $(yq -o=json "$MATRIX_FILE" | jq -c -r "$MATRIX_QUERY | map(. as \$o | {std: .std[]} + del(\$o.std))")
