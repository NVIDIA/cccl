#!/bin/bash

set -euo pipefail

if [ -z "${GITHUB_ACTIONS:-}" ]; then
  echo "This script must be run in a GitHub Actions environment." >&2
  exit 1
fi

readonly ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly repo_root="$(cd "${ci_dir}/.." && pwd)"

cd "$repo_root"

usage=$(cat <<EOF
Usage: $0 <job_id> <exit code>
EOF
)

if [ "$#" -ne 2 ]; then
  echo "Error: Invalid number of arguments." >&2
  echo "$usage" >&2
  exit 1
fi

job_id="$1"
exit_code="$2"

# Collect workflow-related artifacts -- success state, sccache info, etc.
# These are unpacked and parsed in the workflow-results action.

source ci/util/artifacts/common.sh

# The root of the shared artifact structure:
jobs_artifact_dir="$ARTIFACT_UPLOAD_STAGE/jobs"

# This job's artifact directory:
job_artifacts="$jobs_artifact_dir/$job_id"
mkdir -p "$job_artifacts"

if [[ "$exit_code" -eq 0 ]]; then
  touch "$job_artifacts/success"
fi

# Finds a matching file in the root and copies it to the artifact directory.
find_and_copy_job_artifact_from() {
  filename="$1"
  root="$2"
  filepath="$(find "$root" -name "$filename" -print -quit)"
  if [[ -z "$filepath" ]]; then
    echo "$filename does not exist in repo directory."
    return 1
  fi
  cp -v "$filepath" "$job_artifacts/"
}

find_and_copy_job_artifact_from "sccache_stats.json" build/ || : # Nonfatal if not found

ci/util/artifacts/upload/register.sh "zz_jobs-$job_id" "$jobs_artifact_dir"
