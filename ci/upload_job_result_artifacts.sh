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

# Build diagnostics are uploaded as a separate artifact. The workflow-results action downloads
# the `zz_jobs-*` artifact of every job in the run, so that one only holds what the action reads.
diagnostics_artifact_dir="$ARTIFACT_UPLOAD_STAGE/job_diagnostics"

# This job's diagnostics directory:
job_diagnostics="$diagnostics_artifact_dir/$job_id"

# Finds a matching file in the root and copies it to the artifact directory.
find_and_copy_job_artifact_from() {
  root="$1"
  name="$2"
  if find "$root"/ -maxdepth 4 -name "$name" -type f -printf '' -quit 2>/dev/null; then
    find "$root"/ -maxdepth 4 -name "$name" -type f -print0 | xargs -0 -P4 -I% cp -v % "$job_artifacts"/
  else
    echo "No file matching '$name' found in '$root'."
    return 1
  fi
}

# Finds matching files in the root and copies them to the diagnostics directory, keeping each
# file's path (relative to the repo root for a relative root). The same file name can exist in
# several directories: a job can have one build tree per preset, plus the dependency sub-builds
# under `_deps/`. Copied into a single directory, such files would overwrite each other.
# A leading '.' is dropped from the copied file's name (`.ninja_log` -> `ninja_log`), because
# hidden files are excluded from uploaded artifacts.
find_and_copy_job_diagnostics_from() {
  local root="$1"
  local name="$2"
  local found=false
  local file file_name dest_dir
  while IFS= read -r -d '' file; do
    found=true
    file_name="${file##*/}"
    dest_dir="$job_diagnostics/${file%/*}"
    mkdir -p "$dest_dir"
    cp -v "$file" "$dest_dir/${file_name#.}"
  done < <(find "$root"/ -maxdepth 4 -name "$name" -type f -print0 2>/dev/null)

  if [[ "$found" == false ]]; then
    echo "No file matching '$name' found in '$root'."
    return 1
  fi
}

# The workflow-results action reads this file from the top of the job's artifact directory.
find_and_copy_job_artifact_from build "sccache_stats.json" || : # Nonfatal if not found

find_and_copy_job_diagnostics_from /tmp  "sccache*.log" || : # Nonfatal if not found
find_and_copy_job_diagnostics_from build ".ninja_log"   || : # Nonfatal if not found
find_and_copy_job_diagnostics_from build "build.ninja"  || : # Nonfatal if not found
find_and_copy_job_diagnostics_from build "rules.ninja"  || : # Nonfatal if not found
find_and_copy_job_diagnostics_from build "ctest.log"    || : # Nonfatal if not found

# Skip the diagnostics artifact if no file was copied. Registering it must not fail the job.
if [[ -n "$(find "$job_diagnostics" -type f -print -quit 2>/dev/null)" ]]; then
  ci/util/artifacts/upload/register.sh "zz_diagnostics-$job_id" "$diagnostics_artifact_dir" || :
fi

# Registered last: artifacts are uploaded in this order and a failed upload skips the rest, so
# the job's record is only uploaded if every other artifact was.
ci/util/artifacts/upload/register.sh "zz_jobs-$job_id" "$jobs_artifact_dir"
