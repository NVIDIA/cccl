---
name: cccl-resplit-branch
description: "Rebase a CCCL feature branch onto `main` and resplit its commit history into a clean series, using the same interactive chunk-walkthrough as `cccl-commit`. Backs up the original branch tip, rebases (resolving conflicts), collapses commits to a single working-tree diff via `git reset --mixed`, then hands off to `cccl-commit`'s split / interactive / commit pipeline. Use when a branch has accumulated messy / squashable / out-of-order commits and needs a clean series before opening or refreshing a PR. Trigger phrases: \"resplit this branch\", \"clean up these commits\", \"rebase and resplit\", \"reorganize the commits\", \"squash and resplit\", \"fix up commit history\". For first-time commits on a fresh branch, use `cccl-commit`."
---

# cccl-resplit-branch

Rebase onto `main`, then collapse the branch's commits into a working-tree diff and replay them as a clean series
via `cccl-commit`'s flow. Route every user-facing question through `cccl-clarify`. Refuses on `main`. Never
force-pushes — that's `cccl-pr` Phase 4 with explicit user approval.

## Step 1 — Pre-flight

- Refuse on `main` (`git rev-parse --git-dir` vs `--git-common-dir`).
- Working tree must be clean: `git status --porcelain` empty. Dirty → route to `cccl-commit` first.
- Scratch: `mkdir -p /tmp/claude/<sessionid>`.
- `git log --oneline main..HEAD > /tmp/claude/<sessionid>/original-commits.txt`. Empty → nothing to resplit;
  exit. Branch is already pushed with review activity → `cccl-clarify` confirms the user wants to rewrite
  published history (force-push will come later via `cccl-pr` Phase 4).

## Step 2 — Backup the tip

`cccl-clarify` confirms the backup ref name (default `refs/backup/<branch>-<YYYYMMDD-HHMMSS>`). Then:

```
git update-ref refs/backup/<branch>-<timestamp> HEAD
```

Surface the backup ref in every later confirmation prompt — recovery is `git reset --hard <ref>`.

## Step 3 — Rebase onto `main`

```
git fetch origin main
git rebase origin/main
```

On conflict, for each conflicted file route through `cccl-clarify`:

- **Resolve manually** — read file, present conflict markers verbatim in chat, suggest resolution, user picks.
- **Take ours** — `git checkout --ours <file>`.
- **Take theirs** — `git checkout --theirs <file>`.
- **Skip commit** — `git rebase --skip` (loses content; only for already-redone work).
- **Abort** — `git rebase --abort`; surface backup ref; exit.

After resolution: `git add <file>` per-file (never bulk-stage), then `git rebase --continue`.

### 3.1 Verify

```
git diff main..HEAD --stat > /tmp/claude/<sessionid>/rebased-diff-stat.txt
```

Compare touched-file set to the pre-rebase commit list. Material mismatch → `cccl-clarify` (continue / inspect /
abort to backup).

## Step 4 — Collapse to working tree

```
git reset --mixed main
```

`--mixed` keeps every change in the working tree, unstaged — the starting state `cccl-commit` expects. **Never
`--hard`** (would discard the work). Mutating; expect prompt; surface the backup ref in the prompt.

Verify: `git diff --stat` must match the rebased diff stat from Step 3.1. Material divergence → STOP.

## Step 5 — Hand off to `cccl-commit`

Run `cccl-commit` from Step 1 onward. Splitting and Committing are implicit (a resplit means at least one new
commit), but offer Interactive (strongly recommended — catches drift the original series hid) and Test gate via
`cccl-clarify`.

Seed the chunk planner from the original commit series (read `original-commits.txt`) — the resplit's job is to
*fix* problems, not invent unrelated structure. Use original commit subjects as starting drafts for the new
messages.

## Step 6 — Final tree check

After the last commit:

```
git diff HEAD refs/backup/<branch>-<timestamp> --stat
```

Non-empty → the new branch diverges from the original. Present the delta via `cccl-clarify`:

- **Expected** (user reverted / edited chunks during walkthrough) — accept.
- **Unexpected** — investigate, or `git reset --hard <backup>` to abort.

Report final tip SHA, commit list, backup ref location, and a force-push reminder if the branch was published.

## Recovery

At any time before commits start landing: `git reset --hard refs/backup/<branch>-<timestamp>` restores the
original tip. After commits land: same command, but the new series is lost; surface this trade-off when the user
asks to abort late.

## Hard prohibitions

- Never `git reset --hard` outside an explicit user-confirmed abort.
- Never force-push — `cccl-pr` Phase 4 owns that with its own approval.
- Never delete a backup ref without per-ref user approval.
- Never `--no-verify`.
- Never co-author / tool-attribution footers.
- Never `git rebase --abort` autonomously — only on explicit user choice.

## Handoff to `cccl-pr`

If the branch was published, the resplit requires a force-push. Route to `cccl-pr` Phase 4 — and note its
current force-push prohibition. Until that's opted-in, the user runs `git push --force-with-lease` by hand.
