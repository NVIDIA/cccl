---
name: cccl-ci
description: "Orientation for CCCL's GitHub Actions CI. Pointers to the sources of truth (`ci/matrix.yaml`, `ci-overview.md`, workflow files) and a map of the moving parts. Use when the user asks how CI works here, where a CI behavior is defined, why a job ran or didn't, or what `[skip-*]` tags exist. Trigger phrases: \"how does CI work\", \"where is X CI defined\", \"why did this job run\", \"explain the matrix\". For TRIAGING a CI failure, use `cccl-triage-pr` or `cccl-triage-nightly` instead."
---

# cccl-ci

## Sources of truth

| Topic                                         | File                                                              |
|-----------------------------------------------|-------------------------------------------------------------------|
| Job matrix (PR / nightly / weekly + override) | `ci/matrix.yaml`                                                  |
| Skip tags, override rules, troubleshooting    | `ci-overview.md`                                                  |
| Workflow entry points                         | `.github/workflows/ci-workflow-{pull-request,nightly,weekly}.yml` |
| `/ok to test` policy + trustees               | `.github/copy-pr-bot.yaml`, `CONTRIBUTING.md` § CI                |
| Per-job runner setup                          | `.github/actions/workflow-run-job-{linux,windows}/`               |
| Matrix expansion → dispatchable jobs          | `.github/actions/workflow-build/` running `build-workflow.py`     |
| Job pruning by changed paths                  | `ci/inspect_changes.py`                                           |
| Result aggregation                            | `.github/actions/workflow-results/`                               |
| Bench-request config                          | `ci/bench.yaml`                                                   |
| Git-bisect cloud dispatch                     | `.github/workflows/git-bisect.yml`                                |

## PR run flow

`ci-workflow-pull-request.yml` → `build-workflow.py` reads `ci/matrix.yaml`. Non-empty `workflows.override` wins;
otherwise `inspect_changes.py` prunes by dirty projects from changed paths. Jobs run through
`workflow-run-job-{linux,windows}/` in a devcontainer. `workflow-results/` aggregates; marks failed if any job
failed OR if override is non-empty.

## Scoping a PR's CI (both block merging)

- **`[skip-*]` tags** on the last commit. Tokens in `ci-overview.md`.
- **`workflows.override` in `ci/matrix.yaml`** — replaces the `pull_request` matrix with a targeted subset:

  ```yaml
  workflows:
    override:
      - {jobs: ['build'], project: 'cudax', ctk: '12.0', std: 'all', cxx: ['msvc14.39', 'gcc10', 'clang14']}
  ```

`cccl-ci-overrides` generates both from failed-job names and/or changed-path lists.

## `/ok to test` policy

Draft PRs need `/ok to test <SHA>` from a maintainer to start CI. Route all such requests through the
`cccl-ok-to-test` agent (SHA-gated).

## Gotchas

- Non-empty `workflows.override` blocks merge. Reset to empty before final merge (don't remove the key).
- Any `[skip-*]` tag blocks merge.
- `ci/bench.yaml` must match `ci/bench.template.yaml` to merge.
- `gh pr view --json statusCheckRollup` returns 100k+ tokens for 500-job PRs. Use `gh pr checks`.
- `gh run view --log-failed` errors mid-run. Use `gh api repos/NVIDIA/cccl/actions/jobs/<JID>/logs`.
