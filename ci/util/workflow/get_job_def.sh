#!/bin/bash

set -euo pipefail

readonly ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../" && pwd)"
source "$ci_dir/util/workflow/common.sh"

readonly usage=$(cat <<EOF
Usage: $0 [job_id]

Prints a json object containing the workflow job definition for the specified job ID.
If no job ID is provided, the \$JOB_ID environment variable is used.
If the job ID does not exist in the workflow, an error is raised.
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

job_obj=$(jq --arg job_id "$job_id" '
  to_entries[]
  | .value
  | (
      (select(has("standalone")) | .standalone[] | select(.id == $job_id)) //
      (select(has("two_stage")) | .two_stage[] | .producers[] | select(.id == $job_id)) //
      (select(has("two_stage")) | .two_stage[] | .consumers[] | select(.id == $job_id))
    )
' "$WORKFLOW_DIR/workflow.json")

if [ -z "$job_obj" ]; then
  echo "Error: No job definition found for job ID '$job_id'." >&2
  exit 1
fi

echo "$job_obj" | jq -r
