#!/usr/bin/env bash

set -euo pipefail

output_dir="$1"
pr_number="${2:-}"
repository="${GITHUB_REPOSITORY:?}"
run_id="${GITHUB_RUN_ID:?}"
mkdir -p "${output_dir}"

if [[ -n "${pr_number}" ]]; then
  gh pr diff "${pr_number}" --repo "${repository}" > "${output_dir}/pr.diff"
fi

gh api --paginate --slurp \
  "repos/${repository}/actions/runs/${run_id}/jobs?filter=latest&per_page=100" \
  | jq '
      ([.[].jobs[]] | unique_by(.id)) as $jobs
      | if ($jobs | length) != (.[0].total_count // 0) then
          error("did not collect every workflow job")
        else
          $jobs
          | map(select(
              .conclusion == "failure"
              or .conclusion == "timed_out"
              or .conclusion == "startup_failure"
              or .conclusion == "action_required"
            ))
        end
    ' \
    > "${output_dir}/jobs.json"

mapfile -t failed_job_ids < <(jq -r '.[].id' "${output_dir}/jobs.json")

for job_id in "${failed_job_ids[@]}"; do
  gh api "repos/${repository}/actions/jobs/${job_id}/logs" \
    > "${output_dir}/job-${job_id}.log"
  test -s "${output_dir}/job-${job_id}.log"
done
