---
name: cccl-triage-nightly
description: "Diagnose failures in the latest scheduled CCCL nightly run on `main` in the CCCL repository. Locates the run, groups failures by toolchain/project, fetches representative logs, summarizes, presents to user, and — on approval — applies fixes against a new branch, opens a draft PR, posts `/ok to test <SHA>`. Use when the user asks to triage, diagnose, or fix nightly CI. Trigger phrases: \"triage the nightly\", \"what failed in nightly\", \"diagnose latest nightly\", \"fix nightly CI\", \"investigate nightly run\"."
argument-hint: "[run-id-or-empty]"
---

# cccl-triage-nightly

Same shape as `cccl-triage-pr`, but starts from a workflow run and ends by opening a fresh draft PR.

Scratch dir, single-Bash discipline, worktree safety, and `cccl-clarify` routing match `cccl-triage-pr`.

## Step 1 — Locate the run

User-supplied run ID wins. Otherwise:

```
gh run list --workflow=ci-workflow-nightly.yml --branch=main --limit=1 --json databaseId,conclusion,createdAt,headSha > /tmp/claude/<sessionid>/nightly_run.json
```

Capture `databaseId` and `headSha`. `conclusion: success` → stop.

## Step 2 — Fetch failures

Dispatch `cccl-fetch-ci-failures` with the run ID.

## Steps 3–7 — Group, fetch logs, summarize, present, diagnose

Identical to `cccl-triage-pr` steps 3–7.

## Step 8 — Ship the fix

No existing branch or PR — open a fresh one.

1. **Worktree safety.** Refuse on `main`. Offer to create a new named branch via `cccl-clarify`.
2. **Apply edits.** Per-file approval via `cccl-clarify`. Offer `gh issue create` for any deferred problems.
3. **Override matrix + skip tags.** Dispatch `cccl-ci-overrides` with `failed_jobs:` (TSV path), `paths:` (edited
   files), `for_workflow: nightly`. Reference the nightly run ID in the override comment; skip tags apply to the
   LAST commit only.
4. **Commit.** Route to `cccl-commit`.
5. **Open PR + `/ok to test`.** Route to `cccl-pr` Phase 1. PR body should reference the nightly run + per-cluster
   diagnosis. Multiple PRs → run Phase 1 once per branch, framed via `cccl-clarify`.
