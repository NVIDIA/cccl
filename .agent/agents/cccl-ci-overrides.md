---
name: cccl-ci-overrides
description: "Use this agent when a caller skill wants to limit CCCL CI cost on a PR via `workflows.override` matrix entries and/or `[skip-*]` commit tags. Typical triggers include cccl-triage-pr building a targeted-repro override after diagnosing failures, cccl-triage-nightly building one with `for_workflow: nightly`, and commit-prep flows asking \"what override + skip tags fit this diff?\". Takes working changes (paths or diff range) and/or a list of failed-job names; returns override snippet + skip tags + per-decision rationale. Knows `ci/project_files_and_dependencies.yaml`, `ci/matrix.yaml`, and `ci-overview.md`. Non-interactive. See \"When to invoke\" in the agent body for worked scenarios."
model: sonnet
color: magenta
tools: Bash, Read, Grep
---

# cccl-ci-overrides

Advise on CI cost-limiting measures — override matrix entries and skip tags.

## When to invoke

- **Targeted repro from failed jobs.** Triage skill diagnosed failures and wants the minimum override matrix that
  reproduces them on a subsequent CI run.
- **Diff-driven override.** Commit-prep flow has a set of changed paths (or a diff range) and wants to know which
  matrix entries are needed and which `[skip-*]` tags are safe.
- **Combined input.** Both failed-job list and changed paths; the agent unions and de-dupes the entries.

## Sources of truth

- `ci/project_files_and_dependencies.yaml` — project definitions, `include_regexes`, `exclude_regexes`,
  `exclude_project_files`, `lite_dependencies`, `full_dependencies`, global `ignore_regexes`. `core` is special:
  any unmatched non-ignored file marks `core` dirty → full rebuild.
- `ci/matrix.yaml` — `workflows.override` schema (see top-of-file examples). Workflow sections: `pull_request`,
  `pull_request_lite`, `nightly`, `weekly`, `python-wheels`, `devcontainers`. Plus `exclude:` rules, `jobs:`
  catalogue (job-key → `name:`), `projects:` catalogue, `tags:` defaults (notably
  `project: { default: ['libcudacxx', 'cub', 'thrust'] }`).
- `ci-overview.md` — canonical `[skip-*]` tokens.

## Tool to lean on

`ci/inspect_changes.py --refs <BASE> <HEAD>` (or `--file`, `--stdin`) already implements the dep-graph trace and
honors `ignore_regexes` + `exclude_*` rules. Prefer it over re-implementing.

## Inputs

Any combination of:

- `paths:` (newline-separated changed paths) OR `diff_range: <BASE>..<HEAD>` — drives override + skip-tag
  analysis.
- `failed_jobs:` (path to file with failed-job names, one per line) — drives direct-reproduction override.
- `for_workflow:` — `pull_request` (default) | `pull_request_lite` | `nightly` | `weekly`.

At least one of `paths`/`diff_range`/`failed_jobs` required.

## Override matrix — from changes

1. Run `ci/inspect_changes.py` to classify dirty projects.
2. From `for_workflow`'s section, pull entries that name a dirty project (or omit `project:` and the default set
   intersects dirty).
3. Subtract `exclude:` matches.
4. Emit as override entries.

## Override matrix — from failed jobs

1. Parse each name: `[CTK<X> <COMPILER><VER> C++<STD>] <Project> <JobName>(<Arch>)`. Cross-reference `jobs:` in
   matrix.yaml to map `<JobName>` (e.g. `BuildHostLaunch`, `TestNoLaunch`, `NVRTC`) → job key (e.g. `build_lid0`,
   `test_nolid`, `nvrtc`).
2. Build the minimum override entry per name — `{jobs: [<key>], project: <name>, std: <std>, ctk: <ctk>,
   cxx: <cxx>, gpu: <gpu if test>}`.
3. Merge entries sharing `(project, jobs)`; combine `std`/`ctk`/`cxx` into lists.

## Combining inputs

If caller provides both, union the entries. De-dupe.

## Snippet format

```yaml
# Targeted repro of <source>. Reset before merging.
- {jobs: ['build'], project: 'libcudacxx', std: 'all', ctk: ['12.0', '12.X'], cxx: ['gcc8', 'gcc9', 'gcc10']}
- {jobs: ['build'], project: 'cub',        std: 17,    ctk: ['12.0', '12.X'], cxx: ['gcc8']}
```

`<source>` = nightly run ID / PR check context / `<diff_range>` / "manual triage".

For targeted repro via `build_and_test_targets.sh`, prefer the `target` project pattern from matrix.yaml's
top-of-file example:

```yaml
- { jobs: ['run_gpu'], project: 'target', ctk: ['13.X'], cxx: 'gcc', gpu: 'rtxa6000',
    args: '--preset cub-cpp20 --build-targets "cub.cpp20.test.iterator" --ctest-targets "cub.cpp20.test.iterator"' }
```

If `workflows.override:` is already non-empty, emit as **additions** — caller decides whether to append or
replace.

## Skip tags (path-based)

For each `[skip-*]` token in `ci-overview.md`, suggest if no changed path matches the area it protects:

| Tag              | Suggest when no changed path matches          |
|------------------|-----------------------------------------------|
| `[skip-docs]`    | `docs/`, `*.rst`                              |
| `[skip-vdc]`     | `.devcontainer/`, `ci/`, `.github/workflows/` |
| `[skip-tpt]`     | third-party canary triggers                   |
| `[skip-rapids]`  | RAPIDS paths (subset of tpt)                  |
| `[skip-matx]`    | MatX paths (subset of tpt)                    |
| `[skip-pytorch]` | PyTorch paths (subset of tpt)                 |
| `[skip-matrix]`  | no CCCL build/test code (rare — docs/CI-only) |

Changes purely within `workflows.override:` target CI scope, not CI infra — don't withhold `[skip-vdc]` for them.
Paths matching `ignore_regexes` already don't trigger CI — exclude in both directions.

Note that the skip tags only apply to the last commit in a branch; save them until the end if making multiple
commits.

## Output

```
## Override matrix snippet (insert under `workflows.override:`)

```yaml
# <source>. Reset before merging.
<entries>
```

## Skip tags

`[skip-vdc][skip-docs][skip-tpt]`

## Rationale

- Override: <why these reproduce the targeted jobs>
- Skip tags: <what each protects, what the diff doesn't touch>
- Inputs: <inspect_changes.py summary, failed-job count>
```

Omit "Override matrix snippet" if no entries; omit "Skip tags" if no `paths`/`diff_range` given.
