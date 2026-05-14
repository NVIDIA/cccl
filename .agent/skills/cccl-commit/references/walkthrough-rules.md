# Walkthrough rules

Used by `cccl-commit` Step 4.

## Per-chunk diff display

For each chunk in planned order:

1. Read `chunks/CC-NN.patch`.
2. Render the diff verbatim in chat as a ` ```diff ` fenced block. Per-hunk headers name the
   file and line range. Never use Bash output for diffs.
3. Pattern dedup is allowed for repeated patterns — show the pattern once, list all other
   occurrences with file:line refs.
4. Suggest improvements (numbered, with file:line refs) or note "No suggested changes".

## Action menu

Present via `cccl-clarify`:

- **Stage as-is** — `git apply --cached chunks/CC-NN.patch`. Verify with
  `git diff --cached --stat`; STOP if the staged file list doesn't match the expected set.
- **Apply suggested edits, re-review** — `Edit`, regenerate diff with `git diff -- <files>`, loop.
- **Apply custom edits, re-review** — user describes changes, `Edit`, loop.
- **Leave unstaged** — defer; move to next chunk.
- **Revert** — `git apply -R chunks/CC-NN.patch` (or `git checkout -- <file>` for whole-file).
- **Discuss** — open conversation; loop back to the action menu when resolved.

## Tracking

Maintain per-group state:

- Current group identifier (CC-NN).
- List of staged chunks.
- List of deferred chunks.
- List of reverted chunks.

Report the state summary at the end of each group before proceeding to Step 5.
