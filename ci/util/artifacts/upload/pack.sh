#!/bin/bash

set -euo pipefail

readonly ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"
source "$ci_dir/util/artifacts/common.sh"

readonly usage=$(cat <<EOF
Usage: $0 <artifact_name>

Packs a staged artifact (created using artifact/stage.sh) into a tar.zst archive.

The archive will be generated from the staged index file in \${ARTIFACT_UPLOAD_STAGE}/<artifact_name> and
saved to \${ARTIFACT_UPLOAD_STAGE}/<artifact_name>/<artifact_name>.tar.zst.

Example Usages:
  - $0 test_artifact
EOF
)

if [ "$#" -ne 1 ]; then
  echo "Error: Invalid number of arguments." >&2
  echo "$usage" >&2
  exit 1
fi

if ! command -v zstd > /dev/null 2>&1; then
  echo "Error: zstd not found." >&2
  exit 1
fi

readonly artifact_name="$1"
readonly artifact_stage_path="${ARTIFACT_UPLOAD_STAGE}/${artifact_name}"
readonly artifact_index_file="$artifact_stage_path/artifact_index.txt"
readonly artifact_cwd_file="$artifact_stage_path/artifact_index_cwd.txt"
readonly artifact_archive="${ARTIFACT_UPLOAD_STAGE}/${artifact_name}/${artifact_name}.tar.zst"

echo "Packing artifact '$artifact_stage_path' into '$artifact_archive'"
echo "Using zstd: `which zstd`"
echo "Pulling artifacts from working directory: $(cat "$artifact_cwd_file")"

tar -cv -C "$(cat "$artifact_cwd_file")" -T "$artifact_index_file" \
  | zstd --compress --threads=0 \
  > "$artifact_archive"
