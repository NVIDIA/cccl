#!/bin/bash

set -euo pipefail

readonly ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"
source "$ci_dir/util/artifacts/common.sh"

readonly usage=$(cat <<EOF
Usage: $0 <artifact_archive> <artifact_path>

Unpacks a tar.bz2 archive into the specified directory.

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

readonly artifact_archive="$1"
readonly artifact_path="$2"

echo "::group::Unpacking artifact from '$artifact_archive' to '$artifact_path'"

# Create the artifact path directory if it doesn't exist
mkdir -p "$artifact_path"

$BZIP2_EXE -dc "$artifact_archive" \
  | tar -xv -C "$artifact_path"

echo "::endgroup::"
