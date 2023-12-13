#!/bin/bash

set -euo pipefail

write_output() {
  local key="$1"
  local value="$2"
  echo "$key=$value" | tee --append "${GITHUB_OUTPUT:-/dev/null}"
}

explode_std_versions() {
  jq -cr 'map(. as $o | {std: $o.std[]} + del($o.std))'
}

explode_libs() {
  jq -cr 'map(. as $o | {lib: $o.lib[]} + del($o.lib))'
}

extract_matrix() {
  local file="$1"
  local type="$2"
  local matrix=$(yq -o=json "$file" | jq -cr ".$type")
  write_output "DEVCONTAINER_VERSION" "$(yq -o json "$file" | jq -cr '.devcontainer_version')"

  local nvcc_full_matrix="$(echo "$matrix" | jq -cr '.nvcc' | explode_std_versions )"
  local per_cuda_compiler_matrix="$(echo "$nvcc_full_matrix" | jq -cr ' group_by(.cuda + .compiler.name) | map({(.[0].cuda + "-" + .[0].compiler.name): .}) | add')"
  write_output "PER_CUDA_COMPILER_MATRIX"  "$per_cuda_compiler_matrix"
  write_output "PER_CUDA_COMPILER_KEYS" "$(echo "$per_cuda_compiler_matrix" | jq -r 'keys | @json')"

  write_output "NVRTC_MATRIX" "$(echo "$matrix" | jq '.nvrtc' | explode_std_versions)"

  local clang_cuda_matrix="$(echo "$matrix" | jq -cr '.["clang-cuda"]' | explode_std_versions | explode_libs)"
  write_output "CLANG_CUDA_MATRIX" "$clang_cuda_matrix"
  write_output "CCCL_INFRA_MATRIX" "$(echo "$matrix" | jq -cr '.["cccl-infra"]' )"
}

main() {
  if [ "$1" == "-v" ]; then
    set -x
    shift
  fi

  if [ $# -ne 2 ] || [ "$2" != "pull_request" ]; then
    echo "Usage: $0 [-v] MATRIX_FILE MATRIX_TYPE"
    echo "  -v            : Enable verbose output"
    echo "  MATRIX_FILE   : The path to the matrix file."
    echo "  MATRIX_TYPE   : The desired matrix. Supported values: 'pull_request'"
    exit 1
  fi

  echo "Input matrix file:" >&2
  cat "$1" >&2
  echo "Matrix Type: $2" >&2

  extract_matrix "$1" "$2"
}

main "$@"
