# cccl-triage — PR mode reference

## Resolve PR

User-supplied PR number wins. Otherwise:

```
gh pr view --json number,title,state,url,headRefName,isDraft,headRefOid > /tmp/claude/<sessionid>/triage/pr_meta.json
```

Capture `number` (for `cccl-ci-fetch-failures`) and `headRefOid` (the commit SHA for
`/ok to test`). If no open PR exists on the current branch, route through `cccl-clarify`
to ask for a PR number.

## Path scoping

When the user asks about failures in specific libraries or paths, filter `failed_jobs.tsv`
before clustering by intersecting the `name` column against the relevant path prefixes
(e.g., `cub/`, `thrust/`, `libcudacxx/`). This is optional — full-matrix triage skips it.

## Ship

After edits are approved and `cccl-ci-overrides` output is accepted:

1. Apply override YAML to `ci/matrix.yaml` `workflows.override`.
2. Add skip tags to the last commit message (via `cccl-commit`).
3. Route to `cccl-commit` to stage and commit.
4. Route to `cccl-pr` Phase 4 to push and post `/ok to test <headRefOid>`.

The PR already exists; no PR creation step.

## Worktree safety

Refuse on `main`. The PR must be on a feature branch. If the working tree is on `main`,
route through `cccl-clarify` — the user likely needs to check out the PR branch first.
