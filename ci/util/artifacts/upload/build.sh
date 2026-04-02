#!/bin/bash

set -euo pipefail

ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"
readonly ci_dir
# shellcheck source=ci/util/artifacts/common.sh
source "$ci_dir/util/artifacts/common.sh"

usage=$(cat <<EOF
Usage: $0 <artifact_name>

Builds a physical tree containing a staged artifact created using artifact/stage.sh / unstage.sh.

The artifact root will be located at \${ARTIFACT_UPLOAD_STAGE}/<artifact_name>/<artifact_name>.
EOF
)
readonly usage

if [[ "$#" -ne 1 ]]; then
  echo "Error: Invalid number of arguments." >&2
  echo "$usage" >&2
  exit 1
fi

readonly artifact_name="$1"
readonly artifact_stage_path="${ARTIFACT_UPLOAD_STAGE}/${artifact_name}"
readonly artifact_index_file="$artifact_stage_path/artifact_index.txt"
readonly artifact_cwd_file="$artifact_stage_path/artifact_index_cwd.txt"
readonly artifact_dir="${ARTIFACT_UPLOAD_STAGE}/${artifact_name}/${artifact_name}"
artifact_cwd="$(cat "$artifact_cwd_file")"
readonly artifact_cwd

mkdir -p "$artifact_dir"

echo "Building artifact '$artifact_name' in '$artifact_dir'"
echo "Pulling artifacts from working directory: $artifact_cwd"

(
  cd "$artifact_cwd"
  while IFS= read -r file; do
    cp -v --parents "$file" "$artifact_dir"
  done < "$artifact_index_file"
)
