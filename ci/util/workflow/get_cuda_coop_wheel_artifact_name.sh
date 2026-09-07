#!/usr/bin/env bash

set -euo pipefail

ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../" && pwd)"
readonly ci_dir

if [[ "$#" -lt 1 || "$#" -gt 2 ]]; then
  echo "Usage: $0 wheel [job_id]" >&2
  exit 1
fi

component="$1"
job_id="${2:-${JOB_ID:-}}"
if [[ "$component" != "wheel" ]]; then
  echo "Error: unknown cuda-coop wheel artifact component '$component'" >&2
  exit 1
fi
if [[ -z "$job_id" ]]; then
  echo "Error: no job ID provided and JOB_ID is not set" >&2
  exit 1
fi

job_def=$("${ci_dir}/util/workflow/get_job_def.sh" "$job_id")
py_version=$(jq -r '.origin.matrix_job.py_version' <<<"$job_def")
host=$(jq -r '.origin.matrix_job.cxx_family' <<<"$job_def")
arch=$(jq -r '.origin.matrix_job.cpu' <<<"$job_def")
project=$(jq -r '.origin.matrix_job.project' <<<"$job_def")

if [[ "$host" == "MSVC" ]]; then
  os="windows"
else
  os="linux"
fi

for tag in "$py_version" "$os" "$arch" "$project"; do
  if [[ -z "$tag" || "$tag" == "null" ]]; then
    echo "Error: incomplete job definition for '$job_id'" >&2
    exit 1
  fi
done

echo "wheel-cuda-coop-${component}-${project}-${os}-${arch}-py${py_version}"
