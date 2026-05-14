---
description: |
  CCCL CI plumbing internals — matrix expansion, inspect_changes, skip-tag and
  override mechanics, copy-pr-bot, result aggregation, workflow-build action,
  devcontainer launch from CI. Auto-loads for deep questions about how the CI
  pipeline works, not just what it produces.
  Triggers: "how does CI work internally", "matrix expansion", "inspect_changes details",
  "build-workflow.py", "skip-tag mechanics", "copy-pr-bot internals".
---

Deep-internals reference for CCCL's CI pipeline. Covers the implementation
layer beneath the user-facing overview in `ci-overview.md`.

## Matrix expansion pipeline

Every CI run (PR, nightly, weekly) starts with a `build-workflow` job on
`ubuntu-latest`. That job calls the `.github/actions/workflow-build` composite
action, which:

1. Optionally runs `ci/inspect_changes.py` to classify which projects are dirty
   (PR mode only).
2. Calls `build-workflow.py ci/matrix.yaml --workflows <name>` with
   `--full-build-projects` and `--lite-build-projects` populated from step 1.
3. Calls `prepare-workflow-dispatch.py` to shape the output into dispatch-ready
   JSON.
4. Exports the result as the `workflow` output and uploads the `workflow/`
   artifact.

The parent workflow fans out into four dispatch-group jobs
(`dispatch-groups-linux-two-stage`, `-windows-two-stage`,
`-linux-standalone`, `-windows-standalone`), each receiving a slice of the
expanded matrix.

See `references/matrix-expansion.md` for `build-workflow.py` internals: tag
explosion, std aliases, two-stage producer/consumer grouping, GUID assignment,
and the override path.

## inspect_changes.py and project scoping

On PRs, `inspect_changes.py` diffs `base_sha..HEAD` using
`ci/project_files_and_dependencies.yaml` to classify every changed file. Files
not matched by any project fall into the `core` bucket; any `core` hit triggers
a full rebuild of all projects.

For non-core files, the script propagates rebuild requirements through the
dependency graph:

- `full_dependencies` — dirty dep → dependent gets `FULL_BUILD` (full workflow)
- `lite_dependencies` — dirty dep → dependent gets `LITE_BUILD` (`_lite`
  workflow variant if one exists, otherwise falls back to full)
- Transitive lite dependencies are computed at config-load time

Outputs are space-separated lists emitted as `FULL_BUILD` and `LITE_BUILD`
GitHub Actions step outputs, then passed to `build-workflow.py` via
`--full-build-projects` / `--lite-build-projects`.

See `references/inspect-changes.md` for the full dependency graph, `core`
semantics, and `ignore_regexes` list.

## Skip-tag and override mechanics

Skip tags are read in the `build-workflow` job's **Export workflow flags** step,
directly from `github.event.head_commit.message`. Each tag sets a boolean output
(`matrix_enabled`, `vdc_enabled`, `docs_enabled`, etc.) that gates downstream
jobs via `if:` conditions.

`[bench-only]` is a composite alias: it sets `matrix_enabled=false`,
`vdc_enabled=false`, `docs_enabled=false`, and enables the bench path.

The override matrix is processed inside `build-workflow.py`: if
`workflows.override` in `ci/matrix.yaml` is non-empty and `--allow-override` is
passed (PR mode only), the override list replaces the requested workflow
entirely. The `workflow-results` action fails the workflow if `override.json`
exists, ensuring overrides block merging.

Tags and overrides can be combined. Skip tags apply at the job-dispatch layer;
the override matrix applies at the matrix-expansion layer.

## copy-pr-bot and `/ok to test`

CCCL uses NVIDIA's `copy-pr-bot` GitHub App. On external PRs, the bot does
nothing until a repository member comments `/ok to test <SHA>`. The bot
verifies the SHA matches the PR head, then copies the branch to a
`pull-request/<N>` prefixed branch. The `pull_request` workflow triggers on
pushes to `pull-request/[0-9]+`, not on the originating PR branch.

Internal contributors with SSH-signed commits skip the bot; signed pushes to
the `pull-request/<N>` branch trigger CI automatically.

`additional_trustees` in `.github/copy-pr-bot.yaml` grants `/ok to test`
permission to listed contributors beyond the default set.

See `references/copy-pr-bot.md` for the full flow and SSH signing setup.

## Per-job runner setup (workflow-run-job-linux)

Each dispatched job runs inside `.github/actions/workflow-run-job-linux`
(Windows: `-windows`). The action:

1. Sets `JOB_CUDA`, `JOB_HOST`, `JOB_IMAGE`, `JOB_RUNNER`, `JOB_ENVIRONMENT`
   as env vars.
2. Fetches AWS credentials for the sccache bucket via OIDC.
3. Launches `.devcontainer/launch.sh --docker --cuda $JOB_CUDA --host
   $JOB_HOST` with `--gpus device=...` when on a GPU runner.
4. Inside the container, `eval "${COMMAND}"` runs the CI script (e.g.
   `"./ci/build_cub.sh" -std "17" -arch "80-real"`).
5. On failure, prints a reproducer block with the exact `launch.sh` invocation.
6. Calls `ci/upload_job_result_artifacts.sh` unconditionally to record
   pass/fail for result aggregation.

## Result aggregation (workflow-results)

After all dispatch groups finish, `verify-workflow` calls
`.github/actions/workflow-results`. It:

1. Downloads the `workflow/` artifact (job manifest from `build-workflow`).
2. Downloads all `zz_jobs-*` artifacts (success/fail records from each job).
3. Runs `verify-job-success.py workflow/job_ids.json` — checks that every job
   ID in the manifest has a corresponding success artifact.
4. Runs `prepare-execution-summary.py` and `final-summary.py` to build the PR
   comment table.
5. Posts a sticky PR comment via `marocchino/sticky-pull-request-comment`.
6. Fails the step if `workflow/override.json` exists (blocks merge when
   override is active).

Nightly/weekly runs additionally post Slack notifications on start and
finish/failure.

## Additional resources

- `references/docs.md` — index of CCCL CI documentation.
- `references/tools.md` — CI plumbing scripts with ownership and cross-references.
- `references/inspect_changes_usage.md` — `ci/inspect_changes.py` CLI interface and examples.
- `references/matrix-expansion.md` — `build-workflow.py` internals: tag
  explosion, two-stage grouping, GUID assignment, override path
- `references/inspect-changes.md` — dependency graph, `core` semantics,
  `ignore_regexes`, full/lite classification rules
- `references/copy-pr-bot.md` — copy-pr-bot flow, `/ok to test` mechanics,
  SSH signing
