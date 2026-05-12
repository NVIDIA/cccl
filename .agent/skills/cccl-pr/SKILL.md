---
name: cccl-pr
description: "Manage CCCL pull requests — open a new draft PR after commits land, edit/comment on an existing PR (title, body, draft↔ready, comments), or push + post `/ok to test` to trigger CI. Detects fork-vs-upstream remote, opens drafts via `gh pr create --draft --repo NVIDIA/cccl`, dispatches `cccl-ok-to-test` for SHA-verified CI triggers. Use when pushing a branch and opening a PR, editing an existing PR's title/body, toggling draft/ready, commenting, or triggering CI. Trigger phrases: \"open a PR\", \"push and PR\", \"update PR description\", \"mark PR ready\", \"comment on the PR\", \"trigger CI on PR\". For commits, route to `cccl-commit` first."
---

# cccl-pr

CCCL PR lifecycle. Route every user-facing question through `cccl-clarify`. Refuses on `main`. Never force-pushes;
never deletes branches; never closes/merges PRs.

## Step 1 — Resolve mode

`cccl-clarify` (or infer from phrasing):

- **Open new draft PR** → Phase 1.
- **Edit existing PR** (title / body / draft↔ready / base) → Phase 2.
- **Comment** → Phase 3.
- **Push + `/ok to test`** → Phase 4.

## Phase 1 — Open a new draft PR

### 1.1 Sanity checks

- Refuse on `main` (`git rev-parse --git-dir` vs `--git-common-dir`).
- Refuse if `git status --porcelain` is dirty (route to `cccl-commit`).
- Confirm commits ahead: `git log --oneline origin/main..HEAD`.

### 1.2 Detect push remote

```
gh auth status
git remote -v
gh pr view --json headRepositoryOwner   # if branch already has an upstream PR
```

Fork remote present → push there. Only `origin` and it points at `NVIDIA/cccl` → user is a maintainer; confirm
before pushing. Ambiguous → `cccl-clarify`.

### 1.3 Push

`git push -u <remote> <branch>` (mutating; expect prompt). Capture any "view PR" URL hint from the output.

### 1.4 Draft title + body, open PR

Seed from `git log --oneline main..HEAD`. Title ≤ 72 chars, imperative. Body: bulleted commit summary, refs to
issues/PRs, test plan when non-trivial. `cccl-clarify` → confirm / revise / cancel.

Print the generated PR description to chat and ask the user to confirm or edit. On confirm, write to `/tmp/claude/<sessionid>/pr-body.md` and run:

```
gh pr create --draft --repo NVIDIA/cccl --base main \
  --head <fork-owner>:<branch> \
  --title "<title>" \
  --body-file /tmp/claude/<sessionid>/pr-body.md
```

Capture the new PR number from the returned URL.

### 1.5 Trigger CI

`cccl-clarify` → dispatch `cccl-ok-to-test` now (recommended; drafts need `/ok to test <SHA>` to start CI). Then
suggest `ScheduleWakeup(delaySeconds=1200)` polling on `gh pr checks <PR#>`.

## Phase 2 — Edit an existing PR

Resolve PR# from current branch (`gh pr view --json number`) or user input. `cccl-clarify`:

- **Edit title** — draft, confirm, `gh pr edit <PR#> --title "<new>"`.
- **Edit body** — read current via `gh pr view <PR#> --json body`, draft, confirm,
  `gh pr edit <PR#> --body-file /tmp/claude/<sessionid>/pr-body.md`.
- **Mark ready** — `gh pr ready <PR#>`.
- **Mark draft** — `gh pr ready <PR#> --undo`.
- **Change base** — `gh pr edit <PR#> --base <new-base>`. Rare.

All mutating; one approval per use, never bundled.

## Phase 3 — Comment

Resolve PR#. Draft body, confirm via `cccl-clarify`, then:

```
gh pr comment <PR#> --repo NVIDIA/cccl --body "<comment>"
```

For `/ok to test <SHA>` specifically, use Phase 4 — the `cccl-ok-to-test` agent owns the SHA gate.

## Phase 4 — Push + `/ok to test`

For an existing PR whose branch has new local commits.

1. `git push <remote> <branch>` (never force unless *explicitly* told by the user).
2. Dispatch the `cccl-ok-to-test` agent. It owns the SHA verification, the comment, and the polling reminder.

## Hard prohibitions

- Never force-push (no `--force`, no `+<ref>`).
- Never `gh pr close` / `gh pr merge` — out of scope.
- Never bypass the `cccl-ok-to-test` SHA gate by posting `/ok to test` directly.
- Never edit on `main`.
- Never bundle multiple mutating ops into one approval.
