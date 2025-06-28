#!/bin/bash

set -euo pipefail

readonly usage=$(cat <<EOF
Usage: $0 <run id> <job_id>

Allows the scripts in ci/util/workflow and ci/util/artifacts to run as though they are running in a CI environment.

Create a mock job environment for testing purposes.

The run id can be found in the workflow run URL, and the job_id can be found at the start of the Run Command job step.

A new shell is spawned with the remote environment of a specific job from a specific workflow run.

Environment variables are configured to mimic the CI environment.

Caches and previously downloaded artifacts in /tmp are deleted to ensure a clean state.
    !! Note that this does affect the caller's filesystem:
       /tmp/workflow
       /tmp/<artifact stages, archives, registry>
       and similar caches will be deleted **from the caller's filesystem**.

       This is usually fine, but be might overwrite files in-use by other mock environments.
EOF
)

if [ "$#" -ne 2 ]; then
  echo "Error: Invalid number of arguments." >&2
  echo "$usage" >&2
  exit 1
fi

if [ -n "${GITHUB_ACTIONS:-}" ]; then
  echo "$0: Detected another GITHUB_ACTIONS environment." >&2
  echo "unset GITHUB_ACTIONS if this is intentional." >&2
  exit 1
fi

if [ -z "${DEVCONTAINER_NAME:-}" ]; then
  echo "This script must be run inside a devcontainer." >&2
  exit 1
else
  echo "Running in devcontainer: $DEVCONTAINER_NAME"
fi

ci_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../" && pwd)

export GITHUB_ACTIONS=true
export GITHUB_RUN_ID="$1"
export JOB_ID="$2"

(
  source "$ci_dir/util/workflow/common.sh"
  source "$ci_dir/util/artifacts/common.sh"

  rm -rf "$WORKFLOW_DIR"
  rm -rf "$ARTIFACT_ARCHIVES"
  rm -rf "$ARTIFACT_UPLOAD_STAGE"
  rm -rf "$ARTIFACT_UPLOAD_REGISTERY"
)

# Configure shell prompt:
export PS0=""
export PS1="<Mock Job: $GITHUB_RUN_ID $JOB_ID> [\u@\h \W]$ "
export PROMPT_COMMAND=""


echo "Starting new shell for emulating Job $JOB_ID in Run $GITHUB_RUN_ID".
echo ""

bash --norc --noprofile -i || :

echo
echo "Exiting mock job environment."
