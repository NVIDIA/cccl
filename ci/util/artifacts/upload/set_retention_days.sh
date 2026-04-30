#!/bin/bash

set -euo pipefail

ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"
readonly ci_dir
# shellcheck source=ci/util/artifacts/common.sh
source "$ci_dir/util/artifacts/common.sh"

usage=$(cat <<EOF
Usage: $0 <artifact_name> <retention_days>

Sets the retention days for an artifact registered for upload.

Example Usage:
  $0 some_huge_temporary_artifact 1
  $0 some_small_useful_output 7
  $0 some_long_term_artifact 30
EOF
)
readonly usage

if [[ "$#" -lt 2 ]]; then
  echo "Error: Missing arguments." >&2
  echo "$usage" >&2
  exit 1
fi

artifact_name="$1"
retention_days="$2"

# Find the artifact entry and update its retention days
jq --arg name "$artifact_name" --argjson retention_days "$retention_days" \
  'map(if .name == $name then .retention_days = $retention_days else . end)' \
  "$ARTIFACT_UPLOAD_REGISTERY" > "$ARTIFACT_UPLOAD_REGISTERY.tmp" && \
  mv "$ARTIFACT_UPLOAD_REGISTERY.tmp" "$ARTIFACT_UPLOAD_REGISTERY"
