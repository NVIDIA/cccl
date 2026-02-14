#!/bin/bash

set -euo pipefail

readonly ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../" && pwd)"
source "$ci_dir/util/workflow/common.sh"

readonly usage=$(cat <<EOF
Usage: $0 [job_id]

Prints the job ID of the associated producer for the specified consumer job ID.
If no job ID is provided, the \$JOB_ID environment variable is used.
If the number of producers for the job is not exactly one, an error is raised.
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

producers=$(jq --arg job_id "$job_id" '
  to_entries[]
  | select(.value.two_stage)
  | .value.two_stage[]
  | select(any(.consumers[]; .id == $job_id))
  | .producers
' "$WORKFLOW_DIR/workflow.json")

producer_count=$(echo "$producers" | jq 'length')
if [ "$producer_count" -ne 1 ]; then
  echo "Error: Expected exactly one producer for job ID '$job_id', but found ${producer_count:-0}." >&2
  exit 1
fi

producer_id=$(echo "$producers" | jq -r '.[0].id')
if [ -z "$producer_id" ]; then
  echo "Error: No producer ID found for job ID '$job_id'." >&2
  exit 1
fi
echo "$producer_id"
