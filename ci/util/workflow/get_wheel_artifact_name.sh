#!/bin/bash

set -euo pipefail

readonly ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../" && pwd)"

readonly usage=$(cat <<EOF
Usage: $0 [job_id]

Get the name of the wheel file that matches the specified job ID's configuration.
If no job ID is provided, the \$JOB_ID environment variable is used.
If the job ID does not exist in the workflow, or is not a python job, an error is raised.
EOF
)

if [ "$#" -gt 1 ]; then
  echo "Error: Too many arguments." >&2
  echo "$usage" >&2
  exit 1
fi

job_id="${1:-${JOB_ID:-}}"

if [ -z "$job_id" ]; then
  echo "Error: No job ID provided and \$JOB_ID is not set." >&2
  echo "$usage" >&2
  exit 1
fi

job_def=$("${ci_dir}/util/workflow/get_job_def.sh" "$job_id")

py_version=$(echo "$job_def" | jq -r '.origin.matrix_job.py_version')
host=$(echo "$job_def" | jq -r '.origin.matrix_job.cxx_family')
if [[ "$host" == "MSVC" ]]; then
  os="windows"
else
  os="linux"
fi
arch=$(echo "$job_def" | jq -r '.origin.matrix_job.cpu')

for tag in "$py_version" "$os" "$arch"; do
  if [[ -z "$tag" ]]; then
    echo "Error: Missing required field in job definition for job ID '$job_id'." >&2
    echo "$usage" >&2
    echo >&2
    "Job definition: $job_def" >&2
    exit 1
  fi
done

echo "wheel-cccl-$os-$arch-py$py_version"
