---
description: |
  Diagnose and fix CI failures — PR mode or nightly mode. Resolves the run, clusters
  failures, fetches logs, summarizes, and on approval applies fixes plus override matrix
  + skip tags. Triggers: "diagnose the PR", "fix CI on this PR", "what's failing in CI",
  "triage the nightly", "fix nightly CI".
argument-hint: "[PR-number | run-id]"
---

# cccl-triage

Handles both PR-CI and nightly-CI failure triage. The two modes share the same workflow shape
from failure-fetch through diagnosis; they diverge only in how the run is identified and how
the fix is shipped. Route user-question moments through `cccl-clarify`. Create the scratch dir
once: `mkdir -p /tmp/claude/<sessionid>/triage`.

## Step 1 — Identify mode and resolve the run

**PR mode** — triggered by PR context or a user-supplied PR number. See `references/pr.md §Resolve PR`.

**Nightly mode** — triggered by nightly/scheduled CI language or absence of a PR context. See
`references/nightly.md §Locate the run`.

Capture the run's numeric ID and the commit SHA that triggered it. Both are required for Steps 2–4.

## Step 2 — Fetch failures

Dispatch `cccl-ci-fetch-failures` with the resolved run ID. The agent writes a TSV to
`/tmp/claude/<sessionid>/triage/failed_jobs.tsv`: `(job-id, name, grouping-hint)` per row.

Zero failures → report and offer to wait. If waiting, `ScheduleWakeup(delaySeconds=1200)`.

## Step 3 — Cluster + pick representatives

Bucket failures by shared axes (toolchain, library, variant, platform, phase). Pick one
representative job ID per cluster. Full clustering guidance in `references/common.md §Clustering`.

## Step 4 — Pull representative logs

For each representative job ID:

```
gh api repos/NVIDIA/cccl/actions/jobs/<JID>/logs > /tmp/claude/<sessionid>/triage/job_<JID>.log
```

Works mid-run; prefer over `gh run view --log-failed`.

## Step 5 — Summarize

Dispatch one `cccl-ci-summarize-job-log` agent per log, in parallel (haiku tier). Each returns
5–10 lines. Collect summaries to `references/common.md §Log summary format` shape.

## Step 6 — Present findings

Compact table, one row per cluster:

```
Group                             | Repr JID     | Likely cause             | Affected
--------------------------------- | ------------ | ------------------------ | --------
CTK13.2 GCC15 C++20 TestNoLaunch  | 74849038365  | infra: artifact download | 1
CTK12.0 GCC8 C++17 CUB Build      | 74849038xxx  | -Wunused-but-set-param   | 8
```

Route through `cccl-clarify` to ask which clusters to dig into further.

## Step 7 — Diagnose accepted clusters

Re-read representative logs; cross-reference repo code where the error names a file or function.
Per cluster, present:

1. **What broke** — concrete error.
2. **Why** — root-cause hypothesis.
3. **Suggested fix** — concrete change, "rerun — transient infra", or "needs upstream report".
4. **Confidence** — high / medium / low + one-line reason.

For infra-only failures, suggest `gh run rerun <RUN_ID> --failed`.

## Step 8 — Ship the fix

Mode-specific. See `references/pr.md §Ship` or `references/nightly.md §Ship`.

Common to both modes:

- Per-file edits require approval via `cccl-clarify`.
- Dispatch `cccl-ci-overrides` with `failed_jobs:` (TSV path) and `paths:` (edited files).
  Offer the YAML and tag set via `cccl-clarify`. Skip tags apply to the LAST commit only.
- Commit via `cccl-commit`.

## Good-enough criterion

Findings presented to the user with cluster table, per-cluster diagnosis, and a concrete
recommended action. Fix is shipped only on explicit user approval.

## Hard prohibitions

- Never commit or push without explicit per-step user approval.
- Never run on `main` — refuse and offer a branch.
- Never fetch every failure's logs — one representative per cluster.
- Never use `gh pr view --json statusCheckRollup` — 100k+ tokens on large PRs.
- Never use `gh run view --log-failed` mid-run; use the jobs API endpoint.
- Never skip tags except on the last commit of a series.

## Additional resources

- `references/common.md` — clustering axes, log summary format, override-matrix synthesis
- `references/pr.md` — PR-mode run resolution, path scoping, push + `/ok to test`
- `references/nightly.md` — nightly run location, fresh branch/PR creation, for_workflow flag
