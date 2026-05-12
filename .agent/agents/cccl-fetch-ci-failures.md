---
name: cccl-fetch-ci-failures
description: "Use this agent when a caller skill needs the list of failed jobs from a CCCL CI run, given either a PR number or a workflow run ID. Typical triggers include cccl-triage-pr collecting failures for the current branch's PR, cccl-triage-nightly collecting failures for the latest scheduled nightly run, and any other skill that needs failed-job TSV output for downstream summarization or override-matrix generation. Output is a TSV at a caller-specified path with one row per failed job: `<job-id>\\t<full-name>\\t<grouping-hint>`. Handles `gh api --paginate` and the `jq -s` slurp gotcha. Non-interactive. See \"When to invoke\" in the agent body for worked scenarios."
model: haiku
color: cyan
tools: Bash, Read
---

# cccl-fetch-ci-failures

Return failed jobs from a CCCL CI run as TSV.

## When to invoke

- **Triage-PR fetch.** A PR-triage skill has the PR number and needs a TSV of failed jobs to pick representatives
  for log-fetching. Caller hands over PR#, output path, scratch dir.
- **Triage-nightly fetch.** A nightly-triage skill has the workflow run ID (resolved from
  `gh run list --workflow=ci-workflow-nightly.yml`) and needs the same TSV. Caller hands over run ID, output path,
  scratch dir.

## Inputs

One of:
- `pr: <PR#>` — latest run on the PR.
- `run: <RUN_ID>` — specific workflow run.

Plus `output: <path>` and `scratch: <dir>`. Missing any → abort.

## Steps

1. **Resolve the run ID.** If `pr:` given:
   - `gh pr view <PR#> --repo NVIDIA/cccl --json headRefName,headRefOid` → `BRANCH`, `HEAD_SHA`.
   - `gh run list --repo NVIDIA/cccl --branch <BRANCH> --limit 5 --json databaseId,headSha,conclusion` → pick the
     latest entry whose `headSha == HEAD_SHA`. No match → abort.
   - `RUN_ID = databaseId` from that entry.

   Avoid `gh pr view --json statusCheckRollup` — it returns 100k+ tokens on CCCL PRs.
2. **Fetch jobs.** `gh api repos/NVIDIA/cccl/actions/runs/<RUN_ID>/jobs?per_page=100 --paginate` into
   `<scratch>/jobs_raw.json`. `--paginate` concatenates objects; subsequent `jq` needs `-s`.
3. **Extract failures.** `jq -s -r '[.[].jobs[] | select(.conclusion == "failure")] | .[] | [.id, .name] | @tsv'`
   into `<scratch>/failed_jobs_raw.tsv`. Empty → return zero-failures.
4. **Append grouping hints.** Per row, parse the name and append `<toolchain>|<project>|<variant>`:
   - Toolchain: `[CTK<X> <COMPILER><VER> C++<STD>]` substring.
   - Project: CUB / libcudacxx / Thrust / cudax / Python.
   - Variant: Build / Test / HostLaunch / DeviceLaunch / TestNoLaunch / etc.

   Example row:
   ```
   74849038365	[CTK13.2 GCC15 C++20] cudax TestNoLaunch(amd64)	CTK13.2 GCC15 C++20|cudax|TestNoLaunch
   ```

   Write to `<output>`.
5. **Return summary** — count + tally of the third column.
