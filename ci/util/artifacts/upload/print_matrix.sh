#!/bin/bash

set -euo pipefail

ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"
readonly ci_dir
# shellcheck source=ci/util/artifacts/common.sh
source "$ci_dir/util/artifacts/common.sh"

usage=$(cat <<EOF
Prints the Github Actions matrix for uploading all registered artifacts.
EOF
)
readonly usage

if [[ "$#" -ne 0 ]]; then
  echo "$usage" >&2
  exit 1
fi

jq -c '.' "${ARTIFACT_UPLOAD_REGISTERY:?}"
