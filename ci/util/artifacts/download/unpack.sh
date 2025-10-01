#!/bin/bash

set -euo pipefail

readonly ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"
source "$ci_dir/util/artifacts/common.sh"

readonly usage=$(cat <<EOF
Usage: $0 <artifact_name> <artifact_path>

Unpacks a fetched packed artifact's tar.bz2 archive into the specified directory.

Example Usages:
  - $0 /tmp/my_artifact.tar.bz2 /tmp/my_artifact
  - $0 /path/to/archive.tar.bz2 /path/to/extract/
EOF
)

if [ "$#" -lt 2 ]; then
  echo "Error: Missing arguments." >&2
  echo "$usage" >&2
  exit 1
fi

readonly artifact_name="$1"
readonly artifact_path="$2"

readonly artifact_archive="$ARTIFACT_ARCHIVES/${artifact_name}.tar.bz2"

echo "Unpacking artifact from '$artifact_archive' to '$artifact_path'"
echo "Using bzip2 executable: $BZIP2_EXE"

# Create the artifact path directory if it doesn't exist
mkdir -p "$artifact_path"

$BZIP2_EXE -dc "$artifact_archive" \
  | tar -xv -C "$artifact_path"
