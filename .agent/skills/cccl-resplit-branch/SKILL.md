---
description: |
  Rebase a CCCL feature branch onto `main` and resplit its commit history
  into a clean series. Backs up the original tip, rebases (resolving
  conflicts via `cccl-clarify`), collapses to a working-tree diff, then
  hands off to `cccl-commit`. For first-time commits on a fresh branch,
  use `cccl-commit` directly.
  Triggers: "resplit this branch", "clean up these commits", "rebase and resplit", "fix up commit history".
---

# cccl-resplit-branch

Rebase onto `main`, collapse the branch's commits into an unstaged working-tree diff, and replay them as a clean series via `cccl-commit`. Refuses on `main`. Never force-pushes — that is `cccl-pr`'s responsibility.

## Step 1 — Pre-flight

- Refuse on `main`.
- Working tree must be clean (`git status --porcelain` empty). Dirty → route to `cccl-commit` first.
- `mkdir -p /tmp/claude/<sessionid>/resplit`.
- `git log --oneline main..HEAD > /tmp/claude/<sessionid>/resplit/original-commits.txt`. Empty → nothing to resplit; exit.
- Branch is already published with review activity → `cccl-clarify` confirms the user wants to rewrite history (force-push follows later via `cccl-pr`).

## Step 2 — Backup

Confirm backup ref name via `cccl-clarify` (default `refs/backup/<branch>-<YYYYMMDD-HHMMSS>`), then:

```
git update-ref refs/backup/<branch>-<timestamp> HEAD
```

Recovery is possible at any point before final commits land — see [Recovery](#recovery).

## Step 3 — Rebase onto `main`

```
git fetch origin main
git rebase origin/main
```

On conflict, route each file through `cccl-clarify`:

| Choice            | Command                                                                    |
|-------------------|----------------------------------------------------------------------------|
| Resolve manually  | Present conflict markers verbatim; user picks resolution; `git add <file>` |
| Take ours         | `git checkout --ours <file>` then `git add <file>`                       |
| Take theirs       | `git checkout --theirs <file>` then `git add <file>`                     |
| Skip commit       | `git rebase --skip` (loses content — only for already-redone work)        |
| Abort             | `git rebase --abort` then stop; recovery ref still intact                 |

After each resolution: `git add <file>` per-file (never bulk-stage), then `git rebase --continue`.

Verify: `git diff main..HEAD --stat > /tmp/claude/<sessionid>/resplit/rebased-stat.txt`. Material mismatch with the original commit list → `cccl-clarify` (continue / inspect / abort).

## Step 4 — Collapse

```
git reset --mixed main
```

**Never `--hard`** — `--mixed` keeps all changes unstaged, which is the state `cccl-commit` expects. This is irreversible without the backup ref; surface the ref in the confirmation prompt.

Verify: `git diff --stat` must match `rebased-stat.txt`. Divergence → STOP.

## Step 5 — Hand off to `cccl-commit`

Invoke `cccl-commit` from Step 1. Seed the chunk planner from `original-commits.txt` — use original commit subjects as draft message starters. The resplit fixes structure, not invents it.

Offer the Interactive walkthrough (strongly recommended — catches drift the original series hid) and the test gate via `cccl-clarify` before committing.

## Step 6 — Final check

```
git diff HEAD refs/backup/<branch>-<timestamp> --stat
```

Non-empty → present the delta via `cccl-clarify`: expected (user edited chunks) or unexpected (investigate, or reset to backup). Report final tip SHA, commit list, backup ref, and a force-push reminder if the branch was published.

## Recovery

`git reset --hard refs/backup/<branch>-<timestamp>` restores the original tip at any time before the new commits land. After commits land, the same command discards the new series — surface this trade-off explicitly if the user asks to abort late.

## Hard prohibitions

- Never `git reset --hard` without explicit user-confirmed abort.
- Never force-push — `cccl-pr` Phase 4 owns that.
- Never delete a backup ref without per-ref user approval.
- Never `--no-verify`.
- Never `git rebase --abort` autonomously — only on explicit user choice via `cccl-clarify`.

## Handoff

If the branch was published, a force-push is required. Route to `cccl-pr` Phase 4. Until the user opts in, they run `git push --force-with-lease` by hand.
