# cccl-triage — nightly mode reference

## Locate the run

User-supplied run ID wins. Otherwise:

```
gh run list --workflow=ci-workflow-nightly.yml --branch=main --limit=1 --json databaseId,conclusion,createdAt,headSha > /tmp/claude/<sessionid>/triage/nightly_run.json
```

Capture `databaseId` (the run ID for `cccl-ci-fetch-failures`) and `headSha`.
`conclusion: success` → stop and report; nothing to triage.

## Nightly matrix scope

The nightly matrix is broader than the PR matrix — more toolchains, more platforms, more
optional variants. Expect more clusters. When a root cause spans many clusters, confirm
a single fix covers all before generating the override YAML.

Pass `for_workflow: nightly` to `cccl-ci-overrides` so it generates nightly-scoped matrix
overrides. Include a comment in the override YAML referencing the nightly run ID:

```yaml
# Nightly run <databaseId> — <createdAt> — <root-cause summary>
```

## Ship

No existing branch — a new one is required.

1. Route through `cccl-clarify` to name the fix branch (suggest `ci/fix-nightly-<date>`).
2. Offer to create the branch via `cccl-clarify` before applying edits.
3. Apply override YAML to `ci/matrix.yaml` `workflows.override`.
4. Add skip tags to the last commit message (via `cccl-commit`).
5. Route to `cccl-commit` to stage and commit.
6. Route to `cccl-pr` Phase 1 to open a draft PR. PR body must reference:
   - The nightly run ID and timestamp.
   - Per-cluster diagnosis (summary table from Step 6).
   - Any deferred problems (offer `gh issue create` for issues not addressed by the fix).
7. Multiple independent fix branches → run `cccl-pr` Phase 1 once per branch, framed via
   `cccl-clarify`.

## Deferred problems

Failures that need upstream action or are non-trivial to fix in a single PR should be offered
as GitHub issues (`gh issue create`) rather than left as unaddressed clusters in the fix PR.
Route through `cccl-clarify` to confirm before creating issues.
