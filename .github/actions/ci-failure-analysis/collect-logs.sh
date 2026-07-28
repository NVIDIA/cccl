#!/usr/bin/env bash

set -euo pipefail

output_dir="$1"
repository="${GITHUB_REPOSITORY:?}"
run_id="${GITHUB_RUN_ID:?}"
mkdir -p "${output_dir}"

gh api --paginate --slurp \
  "repos/${repository}/actions/runs/${run_id}/jobs?filter=latest&per_page=100" \
  | jq '{
      total_count: (.[0].total_count // 0),
      jobs: ([.[].jobs[]] | unique_by(.id))
    }' \
    > "${output_dir}/jobs.json"

expected_jobs="$(jq '.total_count' "${output_dir}/jobs.json")"
collected_jobs="$(jq '.jobs | length' "${output_dir}/jobs.json")"
if [[ "${collected_jobs}" -ne "${expected_jobs}" ]]; then
  echo "Collected ${collected_jobs} of ${expected_jobs} workflow jobs" >&2
  exit 1
fi

mapfile -t failed_job_ids < <(
  jq -r '
    .jobs[]
    | select(
        .conclusion == "failure"
        or .conclusion == "timed_out"
        or .conclusion == "startup_failure"
        or .conclusion == "action_required"
      )
    | .id
  ' "${output_dir}/jobs.json"
)

for job_id in "${failed_job_ids[@]}"; do
  gh api "repos/${repository}/actions/jobs/${job_id}/logs" \
    > "${output_dir}/job-${job_id}.log"
done
