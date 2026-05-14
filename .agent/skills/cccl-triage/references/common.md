# cccl-triage — common reference

## Clustering

Bucket failures by shared axes before picking representatives. Apply in order:

| Axis      | Examples                                              |
|-----------|-------------------------------------------------------|
| Phase     | configure / build / test / lint / upload              |
| Library   | cub / thrust / libcudacxx / cudax / c                 |
| Toolchain | CTK version + compiler + C++ standard                 |
| Variant   | release / debug / ASAN / sanitizer                    |
| Platform  | linux-amd64 / windows / arm                           |

Pick the representative with the most informative log (build failures > test failures > timeouts).
Infra failures (artifact download, runner setup, network) cluster separately regardless of toolchain.

One representative job ID per cluster. Do not fetch every failure's log.

## Log summary format

Each `cccl-ci-summarize-job-log` dispatch returns 5–10 lines in this shape:

```
STATUS: <PASS|FAIL|INFRA|TIMEOUT|UNKNOWN>
Phase: <configure|build|test|lint|upload>
Error: <verbatim first error line, truncated at 120 chars>
Context: <one sentence — what it was building or testing>
Root cause: <hypothesis — one sentence>
Confidence: <high|medium|low>
Affects: <list of other clusters this root cause likely explains, or "only this cluster">
```

Collect all summaries before presenting Step 6 table. Group identical root-cause hypotheses.

## Override-matrix synthesis

Dispatch `cccl-ci-overrides` with:

- `failed_jobs:` — path to `failed_jobs.tsv`
- `paths:` — list of files touched by the fix
- `for_workflow:` — `pr` (default) or `nightly`

The agent returns a YAML block suitable for `ci/matrix.yaml` `workflows.override` and a set
of `[skip-*]` commit tags. Present both to the user via `cccl-clarify` before applying.

Skip tags apply to the **last** commit of the series only. After CI passes, reset
`workflows.override` to empty and drop the skip tags before merging.

## Pitfalls

- `gh pr checks` exits 1 when any check failed — expected behavior, not an error.
- `gh pr view --json statusCheckRollup` — returns 100k+ tokens on 500-job PRs; avoid.
- `gh run view --log-failed` — unavailable mid-run; use `gh api .../jobs/<JID>/logs` instead.
- A single root cause often explains many clusters (e.g., a new compiler warning). Confirm
  before generating per-cluster fixes.
