#!/bin/bash

set -euo pipefail

ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"
readonly ci_dir
# shellcheck source=ci/util/artifacts/common.sh
source "$ci_dir/util/artifacts/common.sh"

usage=$(cat <<EOF
Usage: $0 <artifact_name> <target_directory>

Downloads files from a named artifact from the current CI workflow run into the specified directory.

Example Usages:
  - $0 my_artifact.tar.gz ./
  - $0 my_artifact /path/to/some/directory/
EOF
)
readonly usage

if [[ "$#" -lt 2 ]]; then
  echo "Error: Missing arguments." >&2
  echo "$usage" >&2
  exit 1
fi

readonly artifact_name="$1"

# Create the target directory and then get its absolute path
mkdir -p "$2"
target_directory="$(cd "$2" && pwd)"
readonly target_directory

echo "Downloading artifact '$artifact_name' to '$target_directory'"
# shellcheck disable=SC2154
"$ci_dir/util/retry.sh" 5 30 \
  gh run download "${GITHUB_RUN_ID}" \
    --name "$artifact_name" \
    --dir "$target_directory"
