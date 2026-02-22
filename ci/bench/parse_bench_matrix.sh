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

Each include entry maps one enabled GPU to a bench_cub.yml workflow invocation.
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

if ! jq -e '.filters? | type == "array" and length > 0 and all(.[]; type == "string")' >/dev/null <<<"${bench_cfg_json}"; then
  die "${bench_yaml_path} must define at least one string entry in benchmarks.filters."
fi

filters_arg="$(
  jq -r '.filters | map(@sh) | join(" ")' <<<"${bench_cfg_json}"
)"

jq -cn \
  --argjson cfg "${bench_cfg_json}" \
  --arg filters "${filters_arg}" \
  '{
    "include": [
      ($cfg.gpus // [])[] as $gpu
      | {
          "gpu": $gpu,
          "launch_args": ($cfg.launch_args // ""),
          "arch": ($cfg.arch // "native"),
          "base_ref": ($cfg.base_ref // "origin/main"),
          "test_ref": ($cfg.test_ref // "HEAD"),
          "filters": $filters,
          "nvbench_args": ($cfg.nvbench_args // ""),
          "nvbench_compare_args": ($cfg.nvbench_compare_args // "")
        }
    ]
  }'
