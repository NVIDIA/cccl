#!/bin/bash

set -euo pipefail

readonly ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"
source "$ci_dir/util/artifacts/common.sh"

readonly usage=$(cat <<EOF
Usage: $0 <artifact_name> <compression_level>

Sets the compression level for an artifact registered for upload.

Example Usage:
  $0 some_huge_precompressed_archive 0
  $0 some_many_small_uncompressed_files 10
EOF
)

if [ "$#" -lt 2 ]; then
  echo "Error: Missing arguments." >&2
  echo "$usage" >&2
  exit 1
fi

artifact_name="$1"
compression_level="$2"

# Find the artifact entry and update its compression level
jq --arg name "$artifact_name" --argjson compression_level "$compression_level" \
  'map(if .name == $name then .compression_level = $compression_level else . end)' \
  "$ARTIFACT_UPLOAD_REGISTERY" > "$ARTIFACT_UPLOAD_REGISTERY.tmp" && \
  mv "$ARTIFACT_UPLOAD_REGISTERY.tmp" "$ARTIFACT_UPLOAD_REGISTERY"
