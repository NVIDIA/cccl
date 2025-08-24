#!/bin/bash

set -euo pipefail

readonly ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"
source "$ci_dir/util/artifacts/common.sh"

readonly usage=$(cat <<EOF
Usage: $0 <artifact_name>

Packs a staged artifact (created using artifact/stage.sh) into a tar.bz2 archive.

The archive will be generated from the staged index file in \${ARTIFACT_UPLOAD_STAGE}/<artifact_name> and
saved to \${ARTIFACT_UPLOAD_STAGE}/<artifact_name>/<artifact_name>.tar.bz2.

Example Usages:
  - $0 test_artifact
EOF
)

if [ "$#" -ne 1 ]; then
  echo "Error: Invalid number of arguments." >&2
  echo "$usage" >&2
  exit 1
fi

readonly artifact_name="$1"
readonly artifact_stage_path="${ARTIFACT_UPLOAD_STAGE}/${artifact_name}"
readonly artifact_index_file="$artifact_stage_path/artifact_index.txt"
readonly artifact_cwd_file="$artifact_stage_path/artifact_index_cwd.txt"
readonly artifact_archive="${ARTIFACT_UPLOAD_STAGE}/${artifact_name}/${artifact_name}.tar.bz2"

echo "Packing artifact '$artifact_stage_path' into '$artifact_archive'"
echo "Using bzip2 executable: $BZIP2_EXE (use pbzip2 for fast parallel compression)"
echo "Pulling artifacts from working directory: $(cat "$artifact_cwd_file")"

tar -cv -C "$(cat ${artifact_cwd_file})" -T "$artifact_index_file" \
  | $BZIP2_EXE -c \
  > "$artifact_archive"
