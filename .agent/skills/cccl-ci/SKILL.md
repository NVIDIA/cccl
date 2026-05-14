---
description: "Orientation for CCCL's GitHub Actions CI: sources of truth, PR run flow, skip tags, override matrix, /ok to test policy, and agent dispatch map. For diagnosing failures, route to cccl-triage instead. Triggers: \"how does CI work\", \"where is X CI defined\", \"why did this job run\", \"explain the matrix\", \"scope this PR's CI\"."
---

# cccl-ci

Sources of truth, flow, and the two mechanisms that scope a PR's CI.

## Sources of truth

| Topic | File |
|-------|------|
| Job matrix (PR / nightly / weekly + override) | `ci/matrix.yaml` |
| Skip tags, override rules, troubleshooting | `ci-overview.md` |
| Workflow entry points | `.github/workflows/ci-workflow-{pull-request,nightly,weekly}.yml` |
| Per-job runner setup | `.github/actions/workflow-run-job-{linux,windows}/` |
| Matrix expansion → dispatchable jobs | `.github/actions/workflow-build/` running `build-workflow.py` |
| Job pruning by changed paths | `ci/inspect_changes.py` |
| Result aggregation | `.github/actions/workflow-results/` |
| Bench-request config | `ci/bench.yaml` |
| Git-bisect cloud dispatch | `.github/workflows/git-bisect.yml` |

## PR run flow

`ci-workflow-pull-request.yml` → `build-workflow.py` reads `ci/matrix.yaml`. Non-empty `workflows.override` wins; otherwise `inspect_changes.py` prunes by dirty projects from changed paths. Jobs run in a devcontainer via `workflow-run-job-{linux,windows}/`. `workflow-results/` aggregates; marks failed if any job failed OR if override is non-empty.

## Scoping a PR's CI (both block merging)

- **`[skip-*]` tags** on the last commit — tokens in `ci-overview.md`.
- **`workflows.override` in `ci/matrix.yaml`** — replaces the `pull_request` matrix with a targeted subset:

  ```yaml
  workflows:
    override:
      - {jobs: ['build'], project: 'cudax', ctk: '12.0', std: 'all', cxx: ['msvc14.39', 'gcc10', 'clang14']}
  ```

`cccl-ci-overrides` generates both from failed-job names and/or changed-path lists.

## /ok to test policy

Draft PRs need `/ok to test <SHA>` from a maintainer to start CI. Route all such requests through `cccl-pr`.

## Agents

| Agent                       | Model  | Purpose                                                                |
|-----------------------------|--------|------------------------------------------------------------------------|
| `cccl-ci-overrides`         | sonnet | Generate `workflows.override` entries and/or `[skip-*]` tags from job names and changed paths |
| `cccl-ci-fetch-failures`    | haiku  | Fetch and list failed jobs for a PR or run                             |
| `cccl-ci-summarize-job-log` | haiku  | Fetch a single job's log and return a structured failure summary       |

## Benchmarks

CI-side benchmark requests are outside this skill's scope. Use `cccl-bench` for writing benchmarks, running the `cccl.bench` tuning harness, and requesting CI bench runs via `ci/bench.yaml`.

## Additional resources

- `references/docs.md` — index of CCCL CI documentation.
- `references/tools.md` — CI-internal scripts with purpose and cross-references.

## Gotchas

- Non-empty `workflows.override` blocks merge. Reset to empty before final merge (don't remove the key).
- Any `[skip-*]` tag on the last commit blocks merge.
- `ci/bench.yaml` must match `ci/bench.template.yaml` to merge.
- `gh pr view --json statusCheckRollup` returns 100k+ tokens for 500-job PRs. Use `gh pr checks` instead.
- `gh run view --log-failed` errors mid-run. Use `gh api repos/NVIDIA/cccl/actions/jobs/<JID>/logs`.
- `gh api --paginate` on a logs endpoint returns a JSON array per page; pipe through `jq -s 'add'` to slurp pages before processing.
