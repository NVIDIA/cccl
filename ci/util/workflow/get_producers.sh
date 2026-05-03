#!/bin/bash

set -euo pipefail

ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../" && pwd)"
readonly ci_dir
# shellcheck source=ci/util/workflow/common.sh
source "$ci_dir/util/workflow/common.sh"

usage=$(cat <<EOF
Usage: $0 [job_id]

Return a json array of job definitions for all producers of the specified consumer job ID.
If no job ID is provided, the \$JOB_ID environment variable is used.
EOF
)
readonly usage

if [[ "$#" -gt 1 ]]; then
  echo "Error: Too many arguments." >&2
  echo "$usage" >&2
  exit 1
fi

job_id="${1:-${JOB_ID:-}}"

if [[ -z "$job_id" ]]; then
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

echo "$producers"
