---
description: "CCCL pull request lifecycle — open a draft PR, edit title/body/state, comment, push new commits, and trigger CI via SHA-verified `/ok to test`. Refuses on `main`; never force-pushes. Triggers: \"open a PR\", \"push and PR\", \"update PR description\", \"mark PR ready\", \"trigger CI on PR\"."
---

# cccl-pr

CCCL PR lifecycle. Route every user-facing question through `cccl-clarify`. Refuses on `main`. Never force-pushes; never deletes branches; never closes or merges PRs.

**Merge-blocker check** — before every push or PR-open, detect non-empty `workflows.override` in `ci/matrix.yaml` and `[skip-*]` tags on HEAD's commit message. Both block merge. Surface via `cccl-clarify` — must be reset before final merge.

## Step 1 — Resolve mode

Infer from phrasing or ask via `cccl-clarify`:

- **Open new draft PR** → Phase 1.
- **Edit existing PR** (title / body / draft↔ready / base) → Phase 2.
- **Comment** → Phase 3.
- **Push + trigger CI** → Phase 4.

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

Fork remote present → push there. Only `origin` pointing at `NVIDIA/cccl` → maintainer; confirm before pushing. Ambiguous → `cccl-clarify`.

### 1.3 Push

`git push -u <remote> <branch>` (mutating; expect prompt).

### 1.4 Draft title + body, open PR

Seed from `git log --oneline main..HEAD`. Title ≤ 72 chars, imperative. Body: bulleted commit summary, issue refs, test plan when non-trivial. Confirm via `cccl-clarify`. On confirm:

```
gh pr create --draft --repo NVIDIA/cccl --base main \
  --head <fork-owner>:<branch> \
  --title "<title>" \
  --body-file /tmp/claude/<sessionid>/pr-body.md
```

Capture the new PR number from the returned URL.

### 1.5 Trigger CI

Ask via `cccl-clarify` whether to trigger CI now. Drafts need `/ok to test <SHA>` to start CI. On yes, run Phase 4 steps.

## Phase 2 — Edit an existing PR

Resolve PR# from current branch (`gh pr view --json number`) or user input. One approval per operation, never bundled.

- **Edit title** — draft, confirm, `gh pr edit <PR#> --title "<new>"`.
- **Edit body** — read current via `gh pr view <PR#> --json body`, draft, confirm, `gh pr edit <PR#> --body-file /tmp/claude/<sessionid>/pr-body.md`.
- **Mark ready** — `gh pr ready <PR#>`.
- **Mark draft** — `gh pr ready <PR#> --undo`.
- **Change base** — `gh pr edit <PR#> --base <new-base>`.

## Phase 3 — Comment

Resolve PR#. Draft body, confirm via `cccl-clarify`, then:

```
gh pr comment <PR#> --repo NVIDIA/cccl --body "<comment>"
```

For `/ok to test <SHA>` specifically, use Phase 4 — it owns the SHA gate.

## Phase 4 — Trigger CI

For a PR whose branch has local commits not yet tested.

### 4.1 Push (if needed)

`git push <remote> <branch>` (never force unless the user explicitly requests it).

Never use `--force` or `+<ref>` unless the user explicitly requests it after seeing the risk.

### 4.2 SHA verification gate

1. `git rev-parse HEAD` → `LOCAL_SHA`.
2. `gh pr view <PR#> --repo NVIDIA/cccl --json headRefOid,isDraft,headRefName` → `REMOTE_SHA`, `isDraft`, `headRefName`.
3. `headRefName` mismatch → abort showing both values.
4. `LOCAL_SHA != REMOTE_SHA` → abort:

```
ERROR: local HEAD does not match remote PR head.
  local:   <LOCAL_SHA>
  remote:  <REMOTE_SHA>
Likely: unpushed commits, or a concurrent push.
Aborting without posting /ok to test.
```

### 4.3 Post comment

```
gh pr comment <PR#> --repo NVIDIA/cccl --body "/ok to test <LOCAL_SHA>"
```

### 4.4 Poll reminder

Suggest `ScheduleWakeup(delaySeconds=1200)` polling on `gh pr checks <PR#>`.

## Good-enough criterion

PR is open, branch is pushed, CI is running (or the user chose to skip triggering CI).

## Hard prohibitions

- Never force-push (`--force`, `+<ref>`).
- Never `gh pr close` or `gh pr merge` — out of scope.
- Never post `/ok to test` without completing the SHA verification gate.
- Never edit on `main`.
- Never bundle multiple mutating operations into one approval.
