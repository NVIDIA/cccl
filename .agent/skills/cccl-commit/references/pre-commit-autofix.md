# Pre-commit autofix flow

Used by `cccl-commit` Step 5.1.

## Install if absent

If `pre-commit` is not on `PATH`, install it into a local venv:

```
python3 -m venv .venv
.venv/bin/pip install pre-commit
```

Then run `.venv/bin/pre-commit run --files <staged>`.

## Auto-fixing hooks

Several hooks modify files in place on failure:

- `pretty-format-json`
- `end-of-file-fixer`
- `trim-trailing-whitespace`
- `ruff format`

When `pre-commit` exits non-zero and the working tree has changed, treat it as an auto-fix run.

## Auto-fix / re-stage flow

1. Show the resulting `git diff` for each modified file.
2. For each file, route through `cccl-clarify`:
   - **Re-stage** — `git apply --cached` the per-file diff.
   - **Revert** — `git checkout -- <file>`.
   - **Discuss** — open conversation; loop.
   Never bulk-`git add` the fixes.
3. Re-run `pre-commit run --files <staged>` to confirm clean.

## Non-auto-fix failures

Hooks that report errors without modifying files (type-checking, lint violations, custom validators):

`cccl-clarify` → investigate and fix / commit anyway / abort.

"Commit anyway" is only appropriate for failures the user understands and accepts; never
suppress with `--no-verify` without explicit user approval at the moment of action.
