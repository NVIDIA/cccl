#!/bin/bash

set -euo pipefail

die() {
  local message="$1"
  local code="${2:-2}"
  echo "${message}" >&2
  exit "${code}"
}

usage() {
  cat <<EOF
Usage: $0 [bench-yaml-path]

Parse ci/bench.yaml and emit a GitHub Actions strategy matrix JSON object:
  {"include":[...]}

Each include entry maps one enabled GPU to a benchmark workflow invocation.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

bench_yaml_path="${1:-ci/bench.yaml}"
if [[ ! -f "${bench_yaml_path}" ]]; then
  die "Benchmark config file not found: ${bench_yaml_path}"
fi

command -v "yq" >/dev/null 2>&1 || die "'yq' is required to parse ${bench_yaml_path}."
command -v "jq" >/dev/null 2>&1 || die "'jq' is required to build the dispatch matrix."

if ! bench_cfg_json="$(yq -o=json '.benchmarks // {}' "${bench_yaml_path}" 2>&1)"; then
  die "Failed to parse ${bench_yaml_path} as YAML: ${bench_cfg_json}"
fi

# Extract CUB and Python filter arrays (default to empty arrays).
cub_filters_json="$(jq -c '.filters.cub // []' <<<"${bench_cfg_json}")"
python_filters_json="$(jq -c '.filters.python // []' <<<"${bench_cfg_json}")"

has_cub_filters="$(jq -e 'type == "array" and length > 0 and all(.[]; type == "string")' <<<"${cub_filters_json}" >/dev/null 2>&1 && echo true || echo false)"
has_python_filters="$(jq -e 'type == "array" and length > 0 and all(.[]; type == "string")' <<<"${python_filters_json}" >/dev/null 2>&1 && echo true || echo false)"

if [[ "${has_cub_filters}" != "true" && "${has_python_filters}" != "true" ]]; then
  die "${bench_yaml_path} must define at least one string entry in benchmarks.filters.cub or benchmarks.filters.python."
fi

cub_filters_arg=""
if [[ "${has_cub_filters}" == "true" ]]; then
  cub_filters_arg="$(jq -r '.filters.cub | map(@sh) | join(" ")' <<<"${bench_cfg_json}")"
fi

python_filters_arg=""
if [[ "${has_python_filters}" == "true" ]]; then
  python_filters_arg="$(jq -r '.filters.python | map(@sh) | join(" ")' <<<"${bench_cfg_json}")"
fi

jq -cn \
  --argjson cfg "${bench_cfg_json}" \
  --arg cub_filters "${cub_filters_arg}" \
  --arg python_filters "${python_filters_arg}" \
  '{
    "include": [
      ($cfg.gpus // [])[] as $gpu
      | {
          "gpu": $gpu,
          "launch_args": ($cfg.launch_args // ""),
          "arch": ($cfg.arch // "native"),
          "base_ref": ($cfg.base_ref // "origin/main"),
          "test_ref": ($cfg.test_ref // "HEAD"),
          "cub_filters": $cub_filters,
          "python_filters": $python_filters,
          "nvbench_args": ($cfg.nvbench_args // ""),
          "nvbench_compare_args": ($cfg.nvbench_compare_args // "")
        }
    ]
  }'
