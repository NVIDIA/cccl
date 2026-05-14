---
description: |
  CCCL's pre-commit hook suite — hooks configured, what each enforces,
  how to install and run locally, the auto-fix-and-restage pattern,
  and how CI enforces the suite via pre-commit.ci.
  Triggers: "run pre-commit", "format code", "lint cccl", "what linters
  run", "clang-format failed", "pre-commit hook failed", "fix formatting".
---

# Pre-commit

Reference and orientation for CCCL's pre-commit setup. Configuration lives
in `.pre-commit-config.yaml`; tool settings (ruff, codespell, mypy) live in
the root `pyproject.toml`; CMake formatter settings are in `.gersemirc`.

## Hook inventory

| Hook                            | Tool                                  | What it checks / fixes                                                                                                                           |
|---------------------------------|---------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|
| `end-of-file-fixer`             | pre-commit/pre-commit-hooks           | Ensures files end with a newline                                                                                                               |
| `mixed-line-ending`             | pre-commit/pre-commit-hooks           | Normalises CRLF → LF                                                                                                                           |
| `trailing-whitespace`           | pre-commit/pre-commit-hooks           | Strips trailing whitespace (non-C/C++/CUDA; those go to clang-format)                                                                         |
| `check-json`                    | pre-commit/pre-commit-hooks           | JSON parse check                                                                                                                               |
| `check-toml`                    | pre-commit/pre-commit-hooks           | TOML parse check                                                                                                                               |
| `pretty-format-json`            | pre-commit/pre-commit-hooks           | Auto-formats JSON: 2-space indent, stable key order                                                                                            |
| `check-symlinks`                | pre-commit/pre-commit-hooks           | Detects broken symlinks                                                                                                                        |
| `check-executables-have-shebangs` | pre-commit/pre-commit-hooks           | Executables must have a shebang                                                                                                                |
| `check-merge-conflict`          | pre-commit/pre-commit-hooks           | Rejects leftover conflict markers                                                                                                              |
| `check-yaml`                    | pre-commit/pre-commit-hooks           | YAML parse check                                                                                                                               |
| `shellcheck`                    | shellcheck-py                         | Shell script linter — excludes `libcudacxx/cmake/config.guess`                                                                                 |
| `clang-format`                  | mirrors-clang-format v20              | Formats `.c/.cpp/.cu/.cuh/.cxx/.h/.hpp/.inl/.mm` and header-only files under `libcudacxx/include/`; uses `.clang-format` (LLVM-based, 120-col limit) |
| `ruff`                          | astral-sh/ruff-pre-commit             | Python linter with auto-fix                                                                                                                    |
| `ruff-format`                   | astral-sh/ruff-pre-commit             | Python formatter                                                                                                                               |
| `gersemi`                       | BlankSpruce/gersemi                   | CMake formatter; 80-col, 2-space indent; custom definitions from `cmake/` and `lib/cmake/thrust/`; extensions in `.gersemi/ext/`              |
| `codespell`                     | codespell-project/codespell           | Spell-check; config in `pyproject.toml [tool.codespell]`; ignore list in `.codespell-ignore.txt`                                              |
| `mypy`                          | pre-commit/mirrors-mypy               | Type-checks `python/cuda_cccl/cuda/compute/` against `python/cuda_cccl/pyproject.toml`; does not run per-file (pass_filenames: false)       |
| `check-shebang`                 | local (`ci/util/pre-commit/check_shebang.py`) | Enforces `#!/usr/bin/env <interp>` form; auto-fixes absolute shebang paths                                                                      |

## Installing locally

```
pip install pre-commit
pre-commit install        # installs the git hook
```

After installation, the suite runs automatically on every `git commit` against
staged files only.

## Running manually

Against staged (or specific) files before committing:

```
pre-commit run --files <file1> <file2> ...
```

Against the entire tree (slow; use for first-time setup or bulk fixes):

```
pre-commit run --all-files
```

## Auto-fix-and-restage

Several hooks modify files in place (clang-format, ruff-format, gersemi,
end-of-file-fixer, pretty-format-json, check-shebang). When a hook rewrites
a file, pre-commit exits non-zero even though the fix was applied.

Pattern:

1. Run pre-commit — it exits non-zero and fixes files.
2. Review the diffs.
3. `git add` the fixed files.
4. Re-run pre-commit (or `git commit` again) — the hooks pass.

Do not skip this review step. Hook auto-fixes occasionally over-correct edge
cases (e.g. clang-format rewrites around CCCL macros; gersemi on hand-formatted
CMake). Inspect the diff before staging.

## CI enforcement

The `.pre-commit-config.yaml` includes a `ci:` block for
[pre-commit.ci](https://pre-commit.ci). This service runs the full hook suite
on every push to a pull-request branch. `autofix_prs: false` means it reports
failures but does not open automatic fix PRs. CI updates hook revisions on a
quarterly schedule.

Linter failures on pre-commit.ci block PR merges. Fix locally and push — the
service re-runs on the next push.

## Skipping a hook

Only when absolutely necessary (e.g., a known false positive in generated code):

```
SKIP=<hook-id> git commit ...
# or: git commit --no-verify  (skips all hooks — use sparingly)
```

Prefer `# noqa`, `# type: ignore`, or a codespell ignore-words entry over
blanket skips. Persistent skips should be encoded in `.pre-commit-config.yaml`
via `exclude:` or `args:` at the hook level.
