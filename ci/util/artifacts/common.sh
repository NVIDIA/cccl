#!/bin/bash

set -euo pipefail

if [ -z "${GITHUB_ACTIONS:-}" ]; then
  echo "This script must be run in a GitHub Actions environment." >&2
  exit 1
fi

readonly ARTIFACT_UPLOAD_STAGE="/tmp/artifact_upload_stage"
readonly ARTIFACT_ARCHIVES="/tmp/artifact_archives"
readonly ARTIFACT_UPLOAD_REGISTERY="$ARTIFACT_ARCHIVES/upload.json"

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
