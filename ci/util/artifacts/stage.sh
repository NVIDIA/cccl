#!/bin/bash

set -euo pipefail

readonly ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../" && pwd)"
source "$ci_dir/util/artifacts/common.sh"

readonly usage=$(cat <<EOF
Usage: $0 <artifact_name> <regex> [<regex> ...]

Stages files matching the provided regexes path for upload under the specified artifact.
Regexes are passed to the 'find' command's -regex option within the artifact stage path and implicitly
start with '^\\./'.

Staged files can be unstaged using the 'artifact/unstage.sh' script.
All stage / unstage operations on the same artifact must be performed from the same working directory.

Once a stage is complete, 'artifact/upload_stage_packed.sh' can be used to create a packed artifact
from the stage. See also 'artifact/upload/pack.sh' and 'artifact/upload/build.sh' for more staging options.

Example Usage:

Stage built binaries and .cmake files in \${ARTIFACT_UPLOAD_STAGE}/test_artifacts for upload:
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

if [[ -f "$artifact_index_cwd" ]]; then
  # Check that the cwd matches the original staging directory if the index already exists:
  if [[ "$(cat "$artifact_index_cwd")" != "$(pwd)" ]]; then
    echo "Error: The current working directory has changed since the artifact was staged." >&2
    echo "Cannot currently stage files from multiple source directories." >&2
    exit 1
  fi
else
  pwd > "$artifact_index_cwd"
fi

echo "Staging artifacts in '$artifact_stage_path'"
for regex in "${regexes[@]}"; do
  # Prepend './' to the regex for convenience. There's an implied ^ at the start of the find regex,
  # and paths always start with ./, so this lets us match top level files directly.
  regex="\\./$regex"
  echo "Staging files matching regex: $regex"
  find . -type f -regex "$regex" | tee -a "$artifact_index_file"
  echo
done
