#!/bin/bash

set -euo pipefail

readonly ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"
source "$ci_dir/util/artifacts/common.sh"

readonly usage=$(cat <<EOF
Usage: $0 <artifact_name> <target_directory>

Downloads files from a named artifact from the current CI workflow run into the specified directory.

Example Usages:
  - $0 my_artifact.tar.gz ./
  - $0 my_artifact /path/to/some/directory/
EOF
)

if [ "$#" -lt 2 ]; then
  echo "Error: Missing arguments." >&2
  echo "$usage" >&2
  exit 1
fi

readonly artifact_name="$1"

# Create the target directory and then get its absolute path
mkdir -p "$2"
readonly target_directory="$(cd "$2" && pwd)"

echo "Downloading artifact '$artifact_name' to '$target_directory'"
"$ci_dir/util/retry.sh" 5 30 \
  gh run download ${GITHUB_RUN_ID} \
    --name "$artifact_name" \
    --dir "$target_directory"
