#!/bin/bash

set -euo pipefail

readonly ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../" && pwd)"
source "$ci_dir/util/workflow/common.sh"

readonly usage=$(cat <<EOF
Usage: $0 [job_id]

Exits successfully if the specified job ID has producers, otherwise exits with an error.
If no job ID is provided, the \$JOB_ID environment variable is used.
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

"${ci_dir}/util/workflow/initialize.sh"

matching_consumer=$(jq --arg job_id "$job_id" '
  to_entries[]
  | select(.value.two_stage)
  | .value.two_stage[]
  | .consumers[]
  | select(.id == $job_id)
' "$WORKFLOW_DIR/workflow.json")

if [ -n "$matching_consumer" ]; then
  exit 0
else
  exit 1
fi
