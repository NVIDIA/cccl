---
name: cccl-commit
description: "Walk uncommitted changes in a CCCL worktree through an interactive review-and-stage flow: survey the diff, optionally split into multiple commit groups, walk chunks one at a time with diff rendering and an action menu (stage / edit / defer / revert), optionally run a test gate, draft commit message(s), confirm, and commit. Use when committing uncommitted changes, preparing a branch for push, or wrapping up a fix. Trigger phrases: \"commit these changes\", \"wrap this up\", \"ready to commit\", \"stage and commit\", \"prepare commits\", \"split into commits\". For PR creation or `/ok to test`, route to `cccl-pr` after committing."
---

# cccl-commit

Interactive commit prep. Route every user-facing question through `cccl-clarify`. Refuses on `main`. Scratch dir:
`mkdir -p /tmp/claude/<sessionid>`.

## Step 1 — Component selection

`AskUserQuestion`, `multiSelect: true`:

- **Split** — group hunks into multiple commits.
- **Interactive** — walk each chunk with a diff render + action menu.
- **Test gate** — run `pre-commit` and a build/test target before committing.
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

Plan into commit groups CC-NN (one group if Split not selected). Within each group, slice into chunks; write each
slice to `/tmp/claude/<sessionid>/chunks/CC-NN.patch`. Coverage check: sum-of-slice-hunks == total-hunks. Run
`git apply --check chunks/CC-NN.patch` on every slice.

Present plan summary (groups, chunks/group, total lines). `cccl-clarify` → approve / reorder / discuss.

## Step 4 — Walk chunks (if Interactive)

For each chunk in planned order:

1. Read `chunks/CC-NN.patch`.
2. Render the diff verbatim in chat as a ` ```diff ` fenced block, per-hunk headers naming file:line range.
   Never use Bash output for diffs. Pattern dedup is fine for repetition — show pattern once, list other
   occurrences and locations.
3. Suggest improvements (numbered, with file:line refs) or note "No suggested changes".
4. `AskUserQuestion`:
   - **Stage as-is** — `git apply --cached chunks/CC-NN.patch`. Verify with `git diff --cached --stat`; STOP if
     the staged file list doesn't match the expected set.
   - **Apply suggested edits, re-review** — `Edit`, regenerate diff with `git diff -- <files>`, loop.
   - **Apply custom edits, re-review** — user describes, `Edit`, loop.
   - **Leave unstaged** — defer.
   - **Revert** — `git apply -R chunks/CC-NN.patch` (or `git checkout -- <file>` for whole-file).
   - **Discuss** — open conversation; loop.

Track: current group, staged/deferred/reverted chunks.

Split selected, Interactive not → auto-stage each slice in order. Verify the staged set grows monotonically into
the per-group expected set. STOP on divergence.

## Step 5 — Test gate (if selected) + commit

### 5.0 Fast path

Commit-only with no Split / no Interactive: confirm staged set via `git diff --cached --stat` (empty → exit),
skip the test gate unless asked, go to 5.2.

### 5.1 Tests

`cccl-clarify` → skip / `pre-commit run --files <staged>` / dispatch `cccl-build-and-test-targets`. If
`pre-commit` is absent, venv-install it (`python3 -m venv .venv && .venv/bin/pip install pre-commit`).

Many pre-commit hooks auto-fix in place (`pretty-format-json`, `end-of-file-fixer`,
`trim-trailing-whitespace`, `ruff format`). On failure with auto-fixes applied:
1. Show the resulting `git diff` per fixed file.
2. For each file, route through `cccl-clarify` — re-stage / revert / discuss — same flow as Step 4's per-chunk
   action menu. Never bulk-`git add` the fixes.
3. Re-run `pre-commit run --files <staged>` to confirm clean.

Other failures: investigate / commit anyway / abort via `cccl-clarify`.

### 5.2 Commit message

`cccl-clarify` for detail tier — **Trivial** (subject only) / **Standard** (subject + 1–6 body lines) /
**Detailed** (subject + multi-paragraph).

Rules:
- Subject ≤ 72 chars, imperative, no trailing period.
- Match CCCL's prefix convention from `git log --oneline -20`.
- Body wraps ~72 chars.
- No co-author / tool-attribution footers.
- `[skip-*]` tags apply to a single push and must appear on the LAST commit's last line only.

Draft. `cccl-clarify` → use / revise / cancel.

### 5.3 Commit

Write final message to `/tmp/claude/<sessionid>/commit-msg-CC.txt`. Then `git commit -F <path>` (mutating; expect
prompt). Verify with `git show -p HEAD`: SHA, subject, file list match expectations.

## Step 6 — Inter-group transition (if Split)

After each commit, `cccl-clarify` → continue / pause / end. On continue, verify remaining slices still apply
(`git apply --check` per remaining slice); regenerate the patch and re-plan if any fail.

Remind caller to use `cccl-ci-overrides` to setup a minimal CI run if needed.

Last group → final summary (all SHAs, deferred, reverted) and exit.

## Hard prohibitions

Unless explicitly approved by the user in `cccl-clarify` at the moment of action, never do any of the following:

- Never edit on `main`.
- Never `--no-verify`.
- Never `--amend` a published commit.
- Never co-author / tool-attribution footers.

In any circumstance:

- Never fabricate diff content — every line shown comes from the patch or `git diff`.
- Never `git add` without explicit per-chunk user approval.

## Handoff

After commits land: route to `cccl-pr` for push / open / update / `/ok to test`.
