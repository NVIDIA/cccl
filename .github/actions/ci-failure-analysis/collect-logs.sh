#!/usr/bin/env bash

set -euo pipefail

output_dir="$1"
run_id="$2"
pr_number="${3:-}"
repository="${GITHUB_REPOSITORY:?}"
mkdir -p "${output_dir}"

if [[ ! "${run_id}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Workflow run ID must be a positive integer" >&2
  exit 1
fi

if [[ -n "${pr_number}" && ! "${pr_number}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Pull request number must be a positive integer" >&2
  exit 1
fi

gh api "repos/${repository}/actions/runs/${run_id}" > "${output_dir}/run.json"

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
