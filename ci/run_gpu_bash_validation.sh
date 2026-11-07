#!/usr/bin/env bash

# Experiment-only replay of a consumer against its completed producer artifacts.
# The parent runner keeps its current job/run IDs for result uploads.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <source-job-id> <shared-ci-script> [args...]" >&2
  exit 2
fi

export GITHUB_RUN_ID=42310798771
export JOB_ID="$1"
shift
echo "Replaying source job ${JOB_ID} from workflow run ${GITHUB_RUN_ID}"
exec bash --noprofile --norc "$@"
