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
  if ! gh pr diff "${pr_number}" --repo "${repository}" > "${output_dir}/pr.diff"; then
    echo "::warning::Unable to collect the pull request diff; continuing without it."
    rm -f "${output_dir}/pr.diff"
  fi
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
          | map({id, name, conclusion})
        end
    ' \
    > "${output_dir}/jobs.json"

mapfile -t failed_job_ids < <(jq -r '.[].id' "${output_dir}/jobs.json")

collected_log_count=0
for job_id in "${failed_job_ids[@]}"; do
  log_path="${output_dir}/job-${job_id}.log"
  if ! gh api --allow-escape-sequences \
    "repos/${repository}/actions/jobs/${job_id}/logs" > "${log_path}"; then
    echo "::warning::Unable to collect logs for job ${job_id}; continuing with job metadata."
    rm -f "${log_path}"
  elif [[ ! -s "${log_path}" ]]; then
    echo "::warning::GitHub returned an empty log for job ${job_id}; continuing with job metadata."
    rm -f "${log_path}"
  else
    collected_log_count=$((collected_log_count + 1))
  fi
done

if [[ "${#failed_job_ids[@]}" -gt 0 && "${collected_log_count}" -eq 0 ]]; then
  echo "::error::Unable to collect logs for any failed workflow jobs."
  exit 1
fi
