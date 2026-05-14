# `ci/inspect_changes.py` usage

Classifies CCCL subprojects as dirty or clean given a set of changed files. Used in the CI
`build-workflow` job to prune the test matrix to only the projects affected by a PR's changes.
Also useful locally to predict which CI jobs a set of edits will trigger.

## Location

`ci/inspect_changes.py`. Run from the repo root (or anywhere — it resolves the repo root from its
own path). Requires Python 3 and `pyyaml`. Config file: `ci/project_files_and_dependencies.yaml`.

## Interface

```
usage: inspect_changes.py [-h] (--refs BASE HEAD | --file PATH | --stdin) [--summary PATH]

Identify which CCCL projects require rebuilds between two commits.

options:
  -h, --help        show this help message and exit
  --refs BASE HEAD  Compare two refs using 'git diff --name-only' to determine dirty files
  --file PATH       Read dirty file paths (one per line) from PATH
  --stdin           Read dirty file paths (one per line) from stdin
  --summary PATH    Optional path to write a markdown summary table
```

## Options

| Flag | Required? | Description |
|------|-----------|-------------|
| `--refs BASE HEAD` | Yes* | Two git refs; diffs from `merge-base(BASE, HEAD)` to `HEAD`. |
| `--file PATH` | Yes* | Newline-separated file of changed paths; bypasses git. |
| `--stdin` | Yes* | Reads changed paths from stdin; bypasses git. |
| `--summary PATH` | No | Writes a Markdown table of project statuses to `PATH`. Used by CI to produce the `workflow/changes.md` section. |

\* Exactly one of `--refs`, `--file`, or `--stdin` is required.

## Output

```
FULL_BUILD=libcudacxx cub thrust
LITE_BUILD=cudax
```

Printed to stdout and, when running inside GitHub Actions, written to `$GITHUB_OUTPUT` as step
outputs. `FULL_BUILD` projects run the full workflow; `LITE_BUILD` projects run the `_lite`
variant (or full as fallback). Empty means no projects need rebuilding.

## Examples

```bash
# Check which projects a PR's changes touch (using the PR's base SHA)
python3 ci/inspect_changes.py --refs origin/main HEAD

# Predict CI impact of a specific set of files
echo "cub/cub/block/block_reduce.cuh" | python3 ci/inspect_changes.py --stdin

# Write a markdown summary alongside stdout output
python3 ci/inspect_changes.py --refs origin/main HEAD --summary /tmp/changes.md

# Check a file list saved by git diff
git diff --name-only origin/main HEAD > /tmp/changed.txt
python3 ci/inspect_changes.py --file /tmp/changed.txt
```

## Notes / gotchas

- Uses the **merge-base** of `BASE` and `HEAD`, not `BASE` itself, to avoid false-positives when
  `HEAD` has drifted behind `origin/main`.
- Shallow repos are automatically unshallowed before diffing.
- Paths matching `ignore_regexes` in `project_files_and_dependencies.yaml` (e.g. `.md` files,
  `docs/`, `.claude/`) are silently excluded — changes to those paths never trigger rebuilds.
- Any file not matched by a named project lands in `core`, which triggers a full rebuild of every
  project. Unknown paths are conservative.
- For dependency propagation details, see `references/inspect-changes.md`.
