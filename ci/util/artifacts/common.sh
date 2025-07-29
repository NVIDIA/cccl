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

export ARTIFACT_UPLOAD_STAGE="/tmp/artifact_upload_stage"
export ARTIFACT_ARCHIVES="/tmp/artifact_archives"
export ARTIFACT_UPLOAD_REGISTERY="$ARTIFACT_UPLOAD_STAGE/artifact_upload_registry.json"

mkdir -p "$ARTIFACT_UPLOAD_STAGE" "$ARTIFACT_ARCHIVES"

if [ ! -f "$ARTIFACT_UPLOAD_REGISTERY" ]; then
  echo "[]" > "$ARTIFACT_UPLOAD_REGISTERY"
fi

# Use parallel bzip2 if available:
if command -v pbzip2 &> /dev/null; then
  BZIP2_EXE=`which pbzip2`
elif command -v bzip2 &> /dev/null; then
  BZIP2_EXE=`which bzip2`
else
  echo "Error: bzip2 or pbzip2 not found." >&2
  exit 1
fi
