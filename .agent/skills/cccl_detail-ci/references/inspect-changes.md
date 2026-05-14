# inspect_changes.py — project change classification

## Source files

- `ci/inspect_changes.py` — main script
- `ci/project_files_and_dependencies.yaml` — project definitions and dep graph

## Classification mechanics

`inspect_changes.py --refs <base_sha> <head_sha>` diffs the two refs with
`git diff --name-only` using the merge-base (not base directly). Shallow repos
are unshallowed first.

Each path is tested against global `ignore_regexes` first. Ignored paths are
tracked separately and do not affect project dirty state.

For surviving paths, each non-`core` project is tested in order:

1. Path matches any `include_regexes` (anchored to repo root).
2. Path does not match any `exclude_regexes`.
3. Path is not in the file set of any `exclude_project_files` project.

A path may match multiple projects.

### core semantics

`core` collects all paths not matched by any non-core project. Any file in
`core` triggers `project_statuses = {all: "Dirty"}` — a full rebuild of every
project with a `matrix_project`. This is the "catch-all" for infra changes.

### Dependency propagation

After initial dirty classification, the script propagates through the reverse
dependency graph:

```
full_dependency edge  → propagate only at depth==0  → FULL_BUILD
lite_dependency edge  → any depth                   → LITE_BUILD
transitive lite dep   → computed at load time        → LITE_BUILD
```

A project in `FULL_BUILD` is tested with the full workflow variant.
A project in `LITE_BUILD` only is tested with the `_lite` workflow variant
(if one exists) or the full workflow as fallback.

Projects without a `matrix_project` field (e.g. `libcudacxx_public`,
`cub_public`, `c2h`) are internal dependency nodes only — they never appear in
`FULL_BUILD`/`LITE_BUILD` outputs. They exist to allow public-API vs
internal-files distinction: a change to `cub/cub/` (public API) propagates
differently than a change to `cub/test/` (internal).

### Output format

```
FULL_BUILD=libcudacxx cub thrust
LITE_BUILD=cudax
```

Written to `$GITHUB_OUTPUT` and also printed to stdout. The `--summary`
optional flag writes a markdown table to a file (used by `action.yml` to
produce the `workflow/changes.md` PR comment section).

## ignore_regexes list (selected entries)

| Pattern                                    | Rationale                                  |
|--------------------------------------------|---------------------------------------------|
| `.+\.md$`                                  | Documentation-only changes never affect build |
| `\.branch_notes/`                          | Local scratch notes, not repo content       |
| `\.claude/`                                | Agent scaffolding                           |
| `docs/`                                    | Sphinx source, not headers or scripts       |
| `ci/bench.*yaml`                           | Bench config, not build/test logic          |
| `.github/workflows/bench.*\.yml`           | Bench workflows                             |
| `.github/workflows/verify-devcontainers\.yml` | VDC workflow                                |

Full list in `ci/project_files_and_dependencies.yaml` under `ignore_regexes`.

## Public/internal split pattern

Projects with large dependency surfaces use a two-key split:

```
cub_public    (include: cub/cub/)          — no matrix_project, dep node only
cub_internal  (include: cub/, exclude: cub_public)  — matrix_project: "cub"
```

A change to `cub/cub/` marks `cub_public` dirty and propagates to all
dependents via `full_dependencies` or `lite_dependencies`. A change to
`cub/test/` marks only `cub_internal` dirty — dependents of `cub_public` are
unaffected.

## Project dependency graph (current)

| Project key                  | Depends on (full)          | Depends on (lite)                |
|------------------------------|---------------------------|----------------------------------|
| `libcudacxx_internal`        | `libcudacxx_public`       | `c2h`                            |
| `cub_internal`               | `cub_public`              | `c2h`, `nvbench_helper`          |
| `thrust_internal`            | `thrust_public`           | `nvbench_helper`                 |
| `cudax_internal`             | `cudax_public`            | `c2h`, `nvbench_helper`          |
| `cccl_c_parallel_internal`   | `cccl_c_parallel_public`  | `c2h`                            |
| `cccl_c_parallel_hostjit`    | `cccl_c_parallel_public`  | `libcudacxx_public`              |
| `python`                     | —                         | `cccl_c_parallel_public`         |
| `tidy`                       | all public+internal keys  | —                                |
| `libcudacxx_public`          | —                         | `thrust_public`, `cub_public`    |
| `cub_public`                 | —                         | `libcudacxx_public`, `thrust_public` |
