#!/bin/bash

set -euo pipefail

readonly ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"
source "$ci_dir/util/artifacts/common.sh"

readonly usage=$(cat <<EOF
Usage: $0 <artifact_path> <artifact_archive>

Packs a staged artifact directory into a tar.bz2 archive.

Example Usages:
  - $0 /tmp/my_artifact /tmp/my_artifact.tar.bz2
EOF
)

if [ "$#" -lt 2 ]; then
  echo "Error: Missing arguments." >&2
  echo "$usage" >&2
  exit 1
fi

readonly artifact_path="$1"
readonly artifact_archive="$2"

echo "::group::Packing artifact '$artifact_path' into '$artifact_archive'"
tar -cv -C "$artifact_path" . \
  | $BZIP2_EXE -c \
  > "$artifact_archive"
echo "::endgroup::"
