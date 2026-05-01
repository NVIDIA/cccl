#!/bin/bash

set -euo pipefail

ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../" && pwd)"
readonly ci_dir
# shellcheck source=ci/util/workflow/common.sh
source "$ci_dir/util/workflow/common.sh"

usage=$(cat <<EOF
Usage: $0

Downloads the workflow artifact and unpacks it to \$WORKFLOW_DIR, but only if it doesn't already exist.
EOF
)
readonly usage

if [[ "$#" -ne 0 ]]; then
  echo "Error: This script does not take any arguments." >&2
  echo "$usage" >&2
  exit 1
fi

if [[ ! -f "$WORKFLOW_DIR/workflow.json" ]]; then
  "$ci_dir/util/artifacts/download/fetch.sh" "$WORKFLOW_ARTIFACT" "$WORKFLOW_DIR" > /dev/null
fi
