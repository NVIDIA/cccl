#!/bin/bash

set -euo pipefail

readonly ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"
source "$ci_dir/util/artifacts/common.sh"

readonly usage=$(cat <<EOF
Usage: $0 <artifact_path> <regex> [<regex> ...]

Stages files matching the provided regexes into the specified artifact path for upload.
Regexes are passed to the `find` command's -regex option in the current directory.
'./' is prepended to all regexes for convenience.

Example Usage:

Stage built binaries in /tmp/build_artifacts for upload:
  $0 /tmp/build_artifacts 'bin/.*' 'lib/.*'

The path can then be registered for upload using:
  $ci_dir/util/artifacts/upload/register.sh <name> /tmp/build_artifacts
EOF
)

if [ "$#" -lt 2 ]; then
  echo "Error: Missing arguments." >&2
  echo "$usage" >&2
  exit 1
fi

stage_path="$1"
shift
regexes=("$@")

# Ensure the stage path is absolute
if [[ "$stage_path" != /* ]]; then
  stage_path="$(pwd)/$stage_path"
fi

mkdir -p "$stage_path"

echo "::group::Staging artifact in '$stage_path'"
for regex in "${regexes[@]}"; do
  regex="./$regex"  # Prepend './' to the regex for convenience
  echo "Staging files matching regex: $regex"
  find . -type f -regex "$regex" -exec cp --parents -v {} "$stage_path" \;
  echo
done
echo "::endgroup::"
