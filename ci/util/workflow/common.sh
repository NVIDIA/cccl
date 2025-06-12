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

export WORKFLOW_ARTIFACT="workflow"
export WORKFLOW_DIR="/tmp/workflow"

mkdir -p "$WORKFLOW_DIR"
