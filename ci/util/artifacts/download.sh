#!/bin/bash

set -euo pipefail

readonly ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../" && pwd)"
source "$ci_dir/util/artifacts/common.sh"

readonly usage=$(cat <<EOF
Usage: $0 <name> [<path>]

Download artifacts uploaded by other jobs in this CI run.

Example Usage:
  Download an artifact to the current directory:
    $0 source_artifact.tar.gz

  Download a packed artifact and extract it to the provided path:
    $0 job-\$ID-products some/path/
EOF
)

if [ "$#" -lt 1 ]; then
  echo "Error: Missing artifact name." >&2
  echo "$usage" >&2
  exit 1
fi
readonly artifact_name="$1"

if [ "$#" -eq 1 ]; then
  artifact_path="./"
else
  artifact_path="$2"
fi

start=$SECONDS
"$ci_dir/util/artifacts/download/fetch.sh" "$artifact_name" "$artifact_path"
echo "Artifact '$artifact_name' downloaded to '$artifact_path' in $((SECONDS - start)) seconds."
