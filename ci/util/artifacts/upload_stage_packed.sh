#!/bin/bash

set -euo pipefail

readonly ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../" && pwd)"
source "$ci_dir/util/artifacts/common.sh"

readonly usage=$(cat <<EOF
Usage: $0 <name>

Same as 'ci/util/artifacts/upload_packed.sh', but assumes that the stage has already been created using
'ci/util/artifacts/stage.sh' and 'unstage.sh'. Performs the packing and registration steps only.
EOF
)

if [ "$#" -ne 1 ]; then
  echo "Error: Invalid number of arguments." >&2
  echo "$usage" >&2
  exit 1
fi

readonly artifact_name="$1"
readonly artifact_archive="$ARTIFACT_UPLOAD_STAGE/${artifact_name}/${artifact_name}.tar.zst"

start=$SECONDS
"$ci_dir/util/artifacts/upload/pack.sh" "$artifact_name"
"$ci_dir/util/artifacts/upload/register.sh" "$artifact_name" "$artifact_archive"
# Already compressed while packing:
"$ci_dir/util/artifacts/upload/set_compression_level.sh" "$artifact_name" 0 > /dev/null
echo "Artifact '$artifact_name' packed in $((SECONDS - start)) seconds."
