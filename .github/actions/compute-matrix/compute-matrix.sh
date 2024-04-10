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

# Filter out the libraries that are dirty
filter_libs() {
  all_libs=("libcudacxx" "thrust" "cub")
  dirty_libs=()
  for lib in "${all_libs[@]}"; do
    dirty_var_name="${lib^^}_DIRTY"
    # If the variable named in dirty_var_name is not set, set it to false:
    : "${!dirty_var_name:=false}"
    # Declare a nameref to the variable named in dirty_var_name
    declare -n lib_dirty="$dirty_var_name"
    # echo "${lib^^}_DIRTY: ${lib_dirty}" >> /dev/stderr
    if [ "${lib_dirty}" = "true" ]; then
      dirty_libs+=("$lib")
    fi
  done
  # echo "Dirty libraries: ${dirty_libs[@]}" >> /dev/stderr

  # Construct a regex to filter out the dirty libraries
  dirty_lib_regex=$(IFS="|"; echo "${dirty_libs[*]}")
  dirty_lib_regex="^(${dirty_lib_regex})\$"
  jq_filter="map(select(.lib | test(\"$dirty_lib_regex\")))"
  jq -cr "$jq_filter"
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

  local clang_cuda_matrix="$(echo "$matrix" | jq -cr '.["clang-cuda"]' | explode_std_versions | explode_libs | filter_libs)"
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
