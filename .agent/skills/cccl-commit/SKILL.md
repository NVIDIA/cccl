---
description: "Interactive commit prep — survey changes, split into commit groups, walk chunks with a diff render and action menu, run a test gate, draft commit messages, and commit. Refuses on `main`. Triggers: \"commit these changes\", \"wrap this up\", \"ready to commit\", \"stage and commit\", \"split into commits\"."
---

# cccl-commit

Interactive commit prep. Route every user-facing question through `cccl-clarify`. Refuses on `main`. Scratch:
`mkdir -p /tmp/claude/<sessionid>`.

## Step 1 — Component selection

`cccl-clarify`, `multiSelect: true`:

- **Split** — group hunks into multiple commits.
- **Interactive** — walk each chunk with a diff render and action menu.
- **Test gate** — run `pre-commit` and/or a build/test target before committing.
- **Commit** — write messages and execute. Without this, nothing commits.

Commit-only with no Split / no Interactive → fast path: commit whatever is staged (Step 5).

## Step 2 — Survey

Single Bash each:

- `git status -sb`
- `git diff > /tmp/claude/<sessionid>/diff-unstaged.txt` (if > 2k lines)
- `git diff --cached > /tmp/claude/<sessionid>/diff-staged.txt` (same threshold)
- `git log --oneline -10`

## Step 3 — Plan (if Split or Interactive)

`git diff > /tmp/claude/<sessionid>/patch.txt` (or `git diff HEAD` for combined).

Plan into commit groups CC-NN. Within each group, slice into chunks; write each slice to
`/tmp/claude/<sessionid>/chunks/CC-NN.patch`. Coverage check: sum-of-slice-hunks == total-hunks.
Run `git apply --check chunks/CC-NN.patch` on every slice.

Present plan summary (groups, chunks/group, total lines). `cccl-clarify` → approve / reorder / discuss.

## Step 4 — Walk chunks (if Interactive)

See `references/walkthrough-rules.md` for diff display rules, the action menu, and tracking.

Split selected, Interactive not → auto-stage each slice in order. Verify the staged set grows
monotonically into the per-group expected set. STOP on divergence.

## Step 5 — Test gate (if selected) + commit

### 5.0 Fast path

No Split / no Interactive: confirm staged set via `git diff --cached --stat` (empty → exit),
skip test gate unless asked, go to 5.2.

### 5.0a Optional CI scoping (last commit only)

Before drafting the last commit message, offer via `cccl-clarify`: scope the next CI run
via `cccl-ci-overrides` — `workflows.override` in `ci/matrix.yaml` and/or `[skip-*]` tags.
Both block merge — remind the user to reset before final merge.

### 5.1 Tests

`cccl-clarify` → skip / `pre-commit run --files <staged>` / dispatch `cccl-test` (or `cccl-build` if a build is needed first).
See `references/pre-commit-autofix.md` for the auto-fix / re-stage flow and edge cases.

### 5.2 Commit message

`cccl-clarify` for detail tier — **Trivial** (subject only) / **Standard** (subject + 1–6 body lines) /
**Detailed** (subject + multi-paragraph). See `references/commit-message-rules.md` for conventions.

Draft. `cccl-clarify` → use / revise / cancel.

### 5.3 Commit

Write final message to `/tmp/claude/<sessionid>/commit-msg-CC.txt`. Then `git commit -F <path>`.
Verify with `git show -p HEAD`: SHA, subject, file list match expectations.

## Step 6 — Inter-group transition (if Split)

After each commit, `cccl-clarify` → continue / pause / end. On continue, verify remaining slices
still apply (`git apply --check` per remaining slice); regenerate and re-plan if any fail.

Last group → final summary (all SHAs, deferred, reverted) and exit.

## Good-enough criterion

All selected commit groups have landed with verified SHAs; no deferred chunks remain unless the user
explicitly left them.

## Hard prohibitions

Unless explicitly approved by the user in `cccl-clarify` at the moment of action:

- Never edit on `main`.
- Never `--no-verify`.
- Never `--amend` a published commit.
- Never co-author / tool-attribution footers.

In any circumstance:

- Never fabricate diff content — every line shown comes from the patch or `git diff`.
- Never `git add` without explicit per-chunk user approval.

## Handoff

After commits land: route to `cccl-pr` for push / open / update / `/ok to test`.

## Additional resources

- `references/walkthrough-rules.md` — per-chunk diff display, action menu, tracking.
- `references/pre-commit-autofix.md` — pre-commit failure, auto-fix detection, re-stage flow.
- `references/commit-message-rules.md` — subject / body conventions, tag prefixes, prohibited content.
