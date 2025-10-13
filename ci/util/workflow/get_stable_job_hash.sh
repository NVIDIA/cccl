#!/bin/bash

set -euo pipefail

readonly ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../" && pwd)"

readonly usage=$(cat <<EOF
Usage: $0 [job_id]

Get a stable hash that identifies the job's toolchain, runner, image,
name, and launch command, removing origin and per-run ids.
If no job ID is provided, the \$JOB_ID environment variable is used.
If the job ID does not exist in the workflow an error is raised.
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

job_def=$("${ci_dir}/util/workflow/get_job_def.sh" "$job_id" | jq 'del(.id, .origin)')
job_hash=$(echo "$job_def" | sha256sum | awk '{print $1}')
echo "$job_hash"
