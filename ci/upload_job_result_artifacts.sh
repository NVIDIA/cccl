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

# The workflow-results action reads this file from the top of the job's directory:
if [[ -d build ]]; then
  find build/ -maxdepth 4 -type f -name "sccache_stats.json" \
    -exec cp -v {} "$job_artifacts"/ \; || : # Nonfatal
fi

# Build diagnostics, for debugging a build after the fact. Nothing in CI reads them.
# They go into one archive so that every file keeps its original path and name: a job can have
# several build trees with same-named files, and hidden files such as `.ninja_log` are excluded
# from uploaded artifacts.
# Every step is nonfatal: a problem here must not keep the job's result from being registered.
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
  # `-P --transform` stores `/tmp/...` as `tmp/...` without tar warning about it in every job log.
  # tar reports an error for a log that is still being written, but still archives it.
  printf '%s\0' "${diagnostics[@]}" \
    | tar --null -T - -P --transform 's,^/,,' -cf - \
    | gzip > "$diagnostics_archive" || :
  # Do not upload an archive that cannot be read or holds nothing (tar or gzip failed):
  [[ -n "$(tar -tz 2>/dev/null < "$diagnostics_archive")" ]] || rm -f "$diagnostics_archive" || :
fi

ci/util/artifacts/upload/register.sh "zz_jobs-$job_id" "$jobs_artifact_dir"
