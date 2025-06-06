#!/bin/bash

set -euo pipefail

readonly ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../" && pwd)"
source "$ci_dir/util/artifacts/common.sh"

readonly usage=$(cat <<EOF
Usage: $0 <artifact_name> <regex> [<regex> ...]

Unstages (removes) files matching the provided regexes from the specified artifact stage.
Regexes follow the same rules as artifact/stage.sh.

This may be used to remove files that were previously staged for upload before packing or building the artifacts.

Example Usage:

Unstage previously-staged built binaries and .cmake files from test_artifacts:
  $0 test_artifacts 'bin/.*' 'lib/.*' '.*cmake$'
EOF
)

if [ "$#" -lt 2 ]; then
  echo "Error: Missing arguments." >&2
  echo "$usage" >&2
  exit 1
fi

artifact_name="$1"
shift
regexes=("$@")

artifact_stage_path="${ARTIFACT_UPLOAD_STAGE}/${artifact_name}"
if [[ "$artifact_stage_path" != /* ]]; then
  artifact_stage_path="$(pwd)/$artifact_stage_path"
fi

mkdir -p "$artifact_stage_path"

artifact_index_file="$artifact_stage_path/artifact_index.txt"
artifact_index_cwd="$artifact_stage_path/artifact_index_cwd.txt"

pwd > "$artifact_index_cwd"

echo "Unstaging artifacts in '$artifact_stage_path'"
for regex in "${regexes[@]}"; do
  # Modify regex for consistency with staging script:
  regex="^\\./$regex"
  echo "Unstaging files matching regex: $regex"
  grep -E "$regex" "$artifact_index_file"
  grep -v -E "$regex" "$artifact_index_file" > "${artifact_index_file}.tmp" && \
    mv "${artifact_index_file}.tmp" "$artifact_index_file"
done
