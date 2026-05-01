#!/bin/bash

set -euo pipefail

ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"
readonly ci_dir
# shellcheck source=ci/util/artifacts/common.sh
source "$ci_dir/util/artifacts/common.sh"

usage=$(cat <<EOF
Usage: $0 <artifact_name> <artifact_path>

Unpacks a fetched packed artifact's tar.zst archive into the specified directory.

Example Usages:
  - $0 /tmp/my_artifact.tar.zst /tmp/my_artifact
  - $0 /path/to/archive.tar.zst /path/to/extract/
EOF
)
readonly usage

if [[ "$#" -lt 2 ]]; then
  echo "Error: Missing arguments." >&2
  echo "$usage" >&2
  exit 1
fi

if ! command -v zstd > /dev/null 2>&1; then
  echo "Error: zstd not found." >&2
  exit 1
fi

readonly artifact_name="$1"
readonly artifact_path="$2"

readonly artifact_archive="$ARTIFACT_ARCHIVES/${artifact_name}.tar.zst"

echo "Unpacking artifact from '$artifact_archive' to '$artifact_path'"
echo "Using zstd executable: $(command -v zstd)"

# Create the artifact path directory if it doesn't exist
mkdir -p "$artifact_path"

zstd --decompress --threads=0 --stdout "$artifact_archive" \
  | tar -xv -C "$artifact_path"
