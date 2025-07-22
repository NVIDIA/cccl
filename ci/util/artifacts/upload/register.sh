#!/bin/bash

set -euo pipefail

readonly ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"
source "$ci_dir/util/artifacts/common.sh"

readonly artifact_compression_level=6
readonly artifact_retention_days=7

readonly usage=$(cat <<EOF
Usage: $0 <artifact_name> [<artifact_path>]

Registers artifacts for upload. If path is not provided, it defaults to the artifact name.

Compression level is set to $artifact_compression_level by default.
Use 'upload/set_compression_level.sh' to change this after registering if needed.

Default retention days is set to $artifact_retention_days.
Use 'upload/set_retention_days.sh' after registering to change this if needed.

Example Usages:
  - $0 my_artifact.tar.gz # Assumes the artifact is in the current directory.
  - $0 my_artifact /path/to/my_artifact.tar.gz
  - $0 my_artifact /path/to/my_artifact_directory/
EOF
)

if [ "$#" -lt 1 ]; then
  echo "Error: Missing artifact name." >&2
  echo "$usage" >&2
  exit 1
fi

artifact_name="$1"
artifact_path="${2:-$artifact_name}"

# Ensure the artifact path is absolute
if [[ "$artifact_path" != /* ]]; then
  artifact_path="$(pwd)/$artifact_path"
fi

if [ ! -e "$artifact_path" ]; then
  echo "Error: Artifact path '$artifact_path' does not exist." >&2
  echo "$usage" >&2
  exit 1
fi

# Register the artifact:
jq --arg name "$artifact_name" \
  --arg path "$artifact_path" \
  --arg retention_days "$artifact_retention_days" \
  --argjson compression_level "$artifact_compression_level" \
  '. += [{"name": $name, "path": $path, "retention_days": ($retention_days | tonumber), "compression_level": $compression_level}]' \
  "$ARTIFACT_UPLOAD_REGISTERY" > "$ARTIFACT_UPLOAD_REGISTERY.tmp" && \
  mv "$ARTIFACT_UPLOAD_REGISTERY.tmp" "$ARTIFACT_UPLOAD_REGISTERY"

echo "Artifact '$artifact_name' registered for upload with path '$artifact_path'."
