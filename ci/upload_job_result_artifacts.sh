#!/usr/bin/env bash

set -euo pipefail

if [[ -z "${GITHUB_ACTIONS:-}" ]]; then
  echo "This script must be run in a GitHub Actions environment." >&2
  exit 1
fi

ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly ci_dir
repo_root="$(cd "${ci_dir}/.." && pwd)"
readonly repo_root

cd "$repo_root"

usage=$(cat <<EOF
Usage: $0 <job_id> <exit code>
EOF
)

if [[ "$#" -ne 2 ]]; then
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

# Read by the workflow-results action from the top of the job's directory:
if [[ -d build ]]; then
  find build/ -maxdepth 4 -type f -name "sccache_stats.json" \
    -exec cp -v {} "$job_artifacts"/ \; || :
fi

# Archive the build diagnostics so that each file keeps its path and name.
diagnostics=()
mapfile -d '' -t diagnostics < <(
  {
    find build/ -maxdepth 4 -type f \
      \( -name ".ninja_log" -o -name "build.ninja" -o -name "rules.ninja" -o -name "ctest.log" \) \
      -print0 || :
    find /tmp/ -maxdepth 4 -type f -name "sccache*.log" -print0 || :
  } 2>/dev/null
) || :

if [[ "${#diagnostics[@]}" -gt 0 ]]; then
  diagnostics_archive="$job_artifacts/diagnostics.tar.gz"
  # -P and --transform store /tmp/... as tmp/... without a warning.
  printf '%s\0' "${diagnostics[@]}" \
    | tar --null -T - -P --transform 's,^/,,' -cf - \
    | gzip > "$diagnostics_archive" || :
  # Remove the archive if it is unreadable or empty:
  if ! members="$(tar -tz 2>/dev/null < "$diagnostics_archive")" || [[ -z "$members" ]]; then
    rm -f "$diagnostics_archive" || :
  fi
fi

ci/util/artifacts/upload/register.sh "zz_jobs-$job_id" "$jobs_artifact_dir"
