---
name: cccl-triage-pr
description: "Diagnose and (optionally) fix CI failures on the current branch's open PR in the CCCL repository. Resolves the PR from the current branch, groups failed checks by likely root cause, pulls representative logs, summarizes them, presents findings, and — on user approval — applies fixes, adds override matrix + skip tags, commits, pushes, posts `/ok to test <SHA>`. Use when the user wants to investigate or fix CI failures on a PR. Trigger phrases: \"diagnose the PR\", \"fix CI on this PR\", \"what's failing in CI\", \"investigate this PR's CI\"."
argument-hint: "[PR-number]"
---

# cccl-triage-pr

Route user-question moments through `cccl-clarify`. Create the scratch dir once: `mkdir -p /tmp/claude/<sessionid>`.

## Step 1 — Resolve PR

User-supplied PR# wins. Otherwise:

```
gh pr view --json number,title,state,url,headRefName,isDraft,headRefOid > /tmp/claude/<sessionid>/pr_meta.json
```

Capture `number` and `headRefOid`.

## Step 2 — Fetch failures

Dispatch `cccl-fetch-ci-failures` with the PR number. The agent writes a TSV to a path you specify
(`/tmp/claude/<sessionid>/failed_jobs.tsv`): `(job-id, name, grouping-hint)` per row.

Zero failures → report and offer to wait. If waiting, schedule `ScheduleWakeup(delaySeconds=1200)`.

## Step 3 — Group + pick representatives

Bucket failures by shared axes (toolchain, library, variant, platform, phase). Pick one representative JID per
group. Don't fetch every failure's logs.

## Step 4 — Pull representative logs

For each representative:

```
gh api repos/NVIDIA/cccl/actions/jobs/<JID>/logs > /tmp/claude/<sessionid>/job_<JID>.log
```

Works mid-run, unlike `gh run view --log-failed`.

## Step 5 — Summarize via `cccl-summarize-job-log`

Dispatch one agent per log, in parallel. Each returns 5–10 lines.

## Step 6 — Present + ask

Compact table:

```
Group                              | Repr JID    | Likely cause             | Affected
---------------------------------- | ----------- | ------------------------ | --------
CTK13.2 GCC15 C++20 TestNoLaunch   | 74849038365 | infra: artifact download | 1
CTK12.0 GCC8 C++17 CUB Build       | 7484903xxxx | -Wunused-but-set-param   | 8
```

Route through `cccl-clarify` to ask which groups to dig into.

## Step 7 — Diagnose accepted groups

Re-read representative logs; cross-reference repo code where the error names a file or function. Present:

1. **What broke** — concrete error.
2. **Why** — root-cause hypothesis.
3. **Suggested fix** — concrete change, "rerun — transient infra", or "needs upstream report".
4. **Confidence** — high/medium/low + one-line reason.

For infra-only failures, suggest `gh run rerun <RUN_ID> --failed`.

## Step 8 — Ship the fix

1. **Worktree safety.** Refuse on `main`.
2. **Apply edits.** Per-file approval via `cccl-clarify`.
3. **Override matrix + skip tags.** Dispatch `cccl-ci-overrides` with `failed_jobs:` (TSV path) + `paths:` (edited
   files). Offer the YAML and tag set via `cccl-clarify`. Skip tags apply to the LAST commit only.
4. **Commit.** Route to `cccl-commit`.
5. **Push + `/ok to test`.** Route to `cccl-pr` Phase 4.

## Pitfalls

- `gh pr checks` exits 1 when any check failed — expected.
- Avoid `gh pr view --json statusCheckRollup` — 100k+ tokens for 500-job PRs.
- Avoid `gh run view --log-failed` mid-run; use `gh api .../jobs/<JID>/logs` instead.
- Don't fetch every failure's logs — one representative per cluster.
