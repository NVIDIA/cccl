#!/bin/bash

set -euo pipefail

readonly ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"
source "$ci_dir/util/artifacts/common.sh"

readonly usage=$(cat <<EOF
Prints the Github Actions matrix for uploading all registered artifacts.
EOF
)

if [ "$#" -ne 0 ]; then
  echo "$usage" >&2
  exit 1
fi

cat $ARTIFACT_UPLOAD_REGISTERY | jq -c '.'
