# CI failure triage

Diagnose the current GitHub Actions run. Retrieve every relevant failure log, group
equivalent actionable failures, inspect the PR changes and source where useful, and
return the supplied JSON schema.

## Retrieve the logs

Use `gh api`, authenticated by `GH_TOKEN`. List every job with one paginated request:

```bash
gh api --paginate --slurp \
  "repos/${GITHUB_REPOSITORY}/actions/runs/${GITHUB_RUN_ID}/jobs?filter=latest&per_page=100" \
  > /tmp/ci-triage-job-pages.json
```

Require the number of unique collected jobs to equal `total_count`. Download the complete
log exactly once for each job concluded as `failure`, `timed_out`, `startup_failure`, or
`action_required`:

```bash
gh api "repos/${GITHUB_REPOSITORY}/actions/jobs/JOB_ID/logs" \
  > /tmp/ci-triage-JOB_ID.log
```

Do not retry, repeat requests, use run-level log endpoints, or wait for the run to finish.
If collection is incomplete or any required log request fails, return
`status: log_retrieval_failed`, briefly explain the error, and leave `jobs`, `groups`, and
`cancelled_job_ids` empty. Do not infer missing logs from source. Ignore the still-running
analysis and publishing jobs.

## Diagnose and group

Start with `git diff "${PR_BASE_SHA}" HEAD`, then use focused, read-only source inspection
to connect failures to the PR or relevant implementation. Do not run builds or tests.

- Use each job's earliest actionable failure; ignore subsequent cleanup, wrapper, and
  aggregation errors.
- Group jobs only when every saved log shows the same decisive signature, causal
  mechanism, and likely remediation. One fix must plausibly resolve the whole group.
- Normalize timestamps, runner paths, generated IDs, and matrix parameters. A shared
  tool, step, or exit code alone does not establish equivalence.
- Put each non-derivative failed job in exactly one group. Omit gate jobs that failed only
  because another job failed, and list cancelled jobs only in `cancelled_job_ids`.
- Keep failures separate when equivalence is uncertain. Order groups by developer value:
  likely PR-caused failures first, then actionable environment or dependency failures.

## Return structured results

Return only one JSON object matching the supplied schema, with no Markdown or commentary.
On success, use `status: ok` and an empty `error`.

For each group:

- `title`: specific failure mechanism, at most 80 characters; never a generic workflow or
  job name.
- `explanation`: one or two sentences describing the failure and developer impact.
- `evidence`: one to three records containing no more than three short log lines total,
  copied verbatim except for ANSI escapes. Each referenced job must belong to the group.
  Use at most one record per job and the job step's numeric `number`, or `0` when
  unavailable.
- `root_cause`: one or two sentences identifying the mechanism; if uncertain, say what
  evidence is missing.
- `source_locations`: up to five supporting repository-relative paths with exact
  one-based line numbers, or an empty array. Do not provide URLs.
- `next_steps`: one or two sentences giving the smallest useful fix or verification,
  including a targeted command when available.
- `agent_prompt`: a self-contained prompt with concrete fix guidance and, when useful, a
  complete proposed fix. Ask the coding agent to verify, reproduce narrowly, implement
  the fix, and run focused validation. Omit repository, run, and job URLs because the
  renderer adds them.
- `job_ids`: every primary job in the group.

In `jobs`, provide the exact numeric ID and name of every job referenced by a group,
evidence record, or `cancelled_job_ids`. Use only observed log lines, IDs, names, step
numbers, source locations, and commands.
