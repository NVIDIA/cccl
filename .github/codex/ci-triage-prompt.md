# CI failure triage

Diagnose the current GitHub Actions run from the collected logs. Group equivalent
actionable failures, inspect the PR changes and source where useful, and return the
supplied JSON schema.

## Evidence

The workflow has already collected the complete job manifest and all relevant failure
logs in `CI_LOG_DIR`:

- `jobs.json` contains `total_count` and the metadata for every job.
- `job-JOB_ID.log` contains the complete log for each job concluded as `failure`,
  `timed_out`, `startup_failure`, or `action_required`.

Read every collected failure log before grouping. Use `jobs.json` for exact job IDs,
names, conclusions, and step numbers. If any required file is missing, return
`status: log_retrieval_failed`, briefly explain the error, and leave `jobs`, `groups`, and
`cancelled_job_ids` empty. Ignore the analysis and publishing jobs.

## Diagnose and group

Inspect `git diff "${PR_BASE_SHA}" HEAD` and relevant source when they can clarify the
failure or connect it to the PR. Do not run builds or tests.

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

- `title`: succinct, specific failure mechanism; never a generic workflow or job name.
- `explanation`: one or two sentences describing the failure and developer impact.
- `evidence`: the decisive log lines, copied verbatim except for ANSI escapes. Each
  referenced job must belong to the group. Use the job step's numeric `number`, or `0`
  when unavailable.
- `root_cause`: one or two sentences identifying the mechanism; if uncertain, say what
  evidence is missing.
- `source_locations`: supporting repository-relative paths with exact one-based line
  numbers, or an empty array. Do not provide URLs.
- `next_steps`: one or two sentences giving the smallest useful fix or verification,
  including a targeted command when available.
- `agent_prompt`: a self-contained prompt with concrete fix guidance and, when useful, a
  complete proposed fix. Ask the coding agent to verify, reproduce narrowly, implement
  the fix, and run focused validation. Omit repository, run, and job URLs because the
  renderer adds them.
- `job_ids`: every primary job in the group.

In `jobs`, provide the exact numeric ID and name of every job referenced by a group,
evidence record, or `cancelled_job_ids`. Do not invent log lines, IDs, names, step
numbers, or source locations.
