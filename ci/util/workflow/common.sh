#!/bin/bash

set -euo pipefail

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "This script must be sourced, not executed directly." >&2
  exit 1
fi

if [ -z "${GITHUB_ACTIONS:-}" ]; then
  echo "This script must be run in a GitHub Actions environment." >&2
  exit 1
fi

to_posix_path() {
  local path="$1"

  if [[ "$path" =~ ^([A-Za-z]):([\\/]?.*)$ ]]; then
    local drive="${BASH_REMATCH[1]}"
    local rest="${BASH_REMATCH[2]}"
    rest="${rest//\\/\/}"
    printf '/%s%s\n' "${drive,,}" "$rest"
    return
  fi

  printf '%s\n' "$path"
}

runner_temp_posix="$(to_posix_path "${RUNNER_TEMP:-/tmp}")"

export WORKFLOW_ARTIFACT="workflow"
export WORKFLOW_DIR="${runner_temp_posix}/workflow"

mkdir -p "$WORKFLOW_DIR"
