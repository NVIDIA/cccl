# CCCL skill & agent framework

## Overview

The `cccl-*` skills and agents wrap CCCL's build, test, CI, benchmarking, commit/PR, and release
infrastructure into named entry points navigated by intent. Top-level skills (`cccl-build`,
`cccl-triage`, `cccl-commit`, `cccl-bench`, `cccl-infra`, …) drive user-facing workflows;
`cccl_detail-*` skills hold shared reference material; read-only agents handle mechanical work like
fetching failed jobs or summarizing logs. Each repeated workflow is encoded once, so every task
starts from a known entry point with relevant project-specific details in context.

---

> **Approval gates remain.** Skills handle the research, drafting, splitting, and
> message composition. Every `git add` / `commit` / `push`, every `gh pr` write
> action, and every `/ok to test` still waits for explicit user approval.

---

## End-to-end prompt examples

> "PR #8965 is failing in CI on the libcudacxx jobs for cuda13.2/gcc14 — figure
> out why, fix it, commit with override tags so we don't re-run the green half of
> the matrix, push, mark ready"

`cccl-triage` (fetch + cluster + summarize) → engineer fix → `cccl-ci-overrides`
(generate the override) → `cccl-commit` (test gate + commit message) →
`cccl-pr` (push + ready + retrigger CI). End-to-end automation of the most
expensive recurring workflow in this repo.

> "device_radix_sort was 1.4x faster on tag 3.0. Bisect, validate the regression
> isn't a SASS-level codegen surprise, fix it, commit, PR, request a bench run."

`cccl-bisect` → `cccl-sass-diff` (validate it's a real algorithmic regression
not codegen drift) → engineer fix → `cccl-bench` (verify locally) →
`cccl-commit` → `cccl-pr` → `cccl-bench` (CI bench request with `[bench-only]`).

> "Resplit this branch — it has 14 messy WIP commits, I want 3 clean ones split by
> library, rebased on current main"

`cccl-resplit-branch` → `cccl-commit`. Backs up tip to `refs/backup/<branch>-<ts>`,
rebases (escalates conflicts via `cccl-clarify`), collapses to working-tree via
`git reset --mixed main`, hands off to `cccl-commit` with the original commit subjects
as starters.

> "I'm onboarding a contributor today. They want to land a small CUB algorithm
> change. Hand them the doc."

`cccl` (entry router) → walks them through: `cccl-devcontainer` → `cccl-cub`
(orientation) → `cccl-build` + `cccl-test` → `cccl-commit` → `cccl-pr`.

---

## 1. Daily inner loop — build, test, iterate

> "Build cub for sm90, then run the device_radix_sort tests"

`cccl-build` → `cccl-test`. Picks the right preset, runs the targeted build, ctest-regexes
the requested suite, reports pass/fail. Fast iteration path, single preset, no matrix.

> "I just touched `cub/cub/device/dispatch/dispatch_reduce.cuh`. Build cub fast and run
> only the device_reduce tests."

`cccl-build` → `cccl-test`. Targeted incremental build via `build_and_test_targets.sh`;
filters CTest by regex.

> "Run the libcudacxx lit tests for `cuda/std/__type_traits/scalar_type.h` under sm90"

`cccl-test`. Picks libcudacxx preset, points lit at the right test directory.

> "Open a shell in a devcontainer with CUDA 13.2 and gcc 14"

`cccl-devcontainer`. Wraps `.devcontainer/launch.sh --cuda 13.2 --host gcc14`.
Detects whether you're already inside a container.

> "Build cudax with the cu13 nightly toolkit in a headless container, then run all
> cudax tests"

`cccl-devcontainer` → `cccl-build` → `cccl-test`. `-d` headless launch with
`-- ./ci/build_cudax.sh` then `./ci/test_cudax.sh`.

> "What CMake presets are available and which one builds everything for native arch?"

`cccl-cmake`. Tabulates presets; recommends `all-dev`.

---

## 2. CI firefighting

> "Triage PR #8963"

`cccl-triage`. Resolves the PR's latest CI run, dispatches `cccl-ci-fetch-failures`
to list failures, clusters by toolchain/library/variant, dispatches
`cccl-ci-summarize-job-log` in parallel (haiku) on representatives, returns a compact
failure-cluster table and asks which clusters to dig into.

> "What's failing on the nightly?"

`cccl-triage` (nightly mode). Same flow, run-id resolved from `nightly.yml`. Especially
useful for the matrix-sized failure sets where you need clustering, not 200 raw logs.

> "Just give me the failed jobs for the current branch -- I want to grep the list myself"

`cccl-ci-fetch-failures` direct. Returns TSV: `<job-id>\t<full-name>\t<grouping-hint>`.

> "Summarize this CI job log: https://github.com/NVIDIA/cccl/actions/runs/.../job/..."

`cccl-ci-summarize-job-log`. Fetches the log, returns failing step, exact command line,
5–20 lines of raw error, and a code/infra/flaky verdict.

> "Generate a `workflows.override` so this PR only re-runs the cub and libcudacxx jobs
> on gcc 14"

`cccl-ci-overrides`. Reads `ci/matrix.yaml` schema, emits the minimum override matrix
snippet plus recommended skip tags, with rationale.

> "Why did the cuda12.6/clang14 job run for this PR? I didn't touch anything that
> needs clang."

`cccl-ci` + `cccl-ci-overrides`. Explains matrix expansion via
`ci/inspect_changes.py` and `project_files_and_dependencies.yaml`, identifies the
trigger path.

> "Walk me through how PR CI is structured — what's the difference between the
> `pull_request` and `nightly` workflows?"

`cccl-ci`. Reference skill — flow diagram, sources of truth, skip-tag mechanics.

---

## 3. Regression hunting

> "device_scan was 1.2x faster a week ago. Find the commit that regressed it."

`cccl-bisect` (cloud route). Dispatches `git-bisect.yml` workflow with the right
runner label, build/test targets, and good/bad refs. Returns the bad commit hash with
the distinguishing command line — a local reproducer.

> "Bisect this segfault on the cuda13.2/gcc14 config — it definitely worked on the
> 3.0 release."

`cccl-bisect`. Resolves `3.0` to a tag, runs cloud bisect, returns the bad commit
with a reproducer command.

> "Bisect locally in a devcontainer — I don't want to wait for the cloud queue"

`cccl-bisect` (local route). Wraps `ci/util/git_bisect.sh` inside
`.devcontainer/launch.sh`.

> "Did my recent CUB tuning change affect codegen for `DeviceRadixSort`?"

`cccl-sass-diff`. Builds both refs, dumps SASS via `cuobjdump`, normalizes addresses
and register renames, reports the top 5 non-trivial diffs by kernel.

---

## 4. Commit / PR endgame

> "Commit these changes"

`cccl-commit`. Component selection → optional split → interactive chunk walkthrough
→ optional test gate → commit message draft (Trivial/Standard/Detailed) → `git commit -F`.
Refuses on `main`.

> "Wrap this up — I want three separate commits split by library (cub, thrust,
> libcudacxx). Run the precommit gate first."

`cccl-commit`. Plans three commit groups, walks chunks, runs pre-commit, drafts per-group
messages, executes each commit.

> "Push and open a draft PR titled `[Tile] Reenable seed_seq tests`"

`cccl-pr` (open new draft). Sanity-check, detect push remote, push branch, open draft PR
with the title and body.

> "Update the PR body to mention the SASS-diff results"

`cccl-pr` (edit existing). `gh pr edit --body-file -`.

> "Mark PR #9001 ready for review"

`cccl-pr` (draft→ready transition).

> "Trigger CI on this PR"

`cccl-pr` (push + trigger). SHA verification gate, then `/ok to test <SHA>` comment.
Never posts without verification.

---

## 5. Library development

> "Add a CUB device-scope algorithm `cub::DeviceMode` that returns the most-frequent
> value. Tour me through the directory layout and tuning policy conventions."

`cccl-cub` (orientation) → manual implementation → `cccl-build` + `cccl-test` to
verify. Covers block/warp/device/agent scopes, the tuning-policy selector pattern,
and Catch2 vs legacy test layout.

> "Make this cudax change libcudacxx-style compliant"

`cccl-libcudacxx` (style references — `headers.md`, `macros.md`, `naming.md`,
`templates.md`, `testing.md`, `visibility.md`). Style enforcement applies to
`libcudacxx/include/` AND `cudax/include/`.

> "Where do I add a new Thrust algorithm with CUDA + cpp + omp + tbb backends?"

`cccl-thrust`. Explains the per-backend directory layout (`thrust/system/{cuda,cpp,omp,tbb}/`),
the ADL dispatch via execution policies, and the typical pattern of `thrust::sort` →
`cub::DeviceRadixSort` for the CUDA backend.

> "What's the C ABI pattern for adding a new algorithm to the C Parallel Library?"

`cccl-c`. Three-call pattern (`_build`, `_run`, `_cleanup`), stable C ABI layer,
JIT-backed cubins via NVRTC, custom iterator/operator types via template strings.

> "What's in cudax that's stable enough to graduate to libcudacxx?"

`cccl-cudax` + `cccl-libcudacxx`. Covers the zero-stability contract and
`CCCL_ENABLE_UNSTABLE` flag on the cudax side; the upstream-tracking model and
where CCCL extensions live on the libcudacxx side.

> "Test `cuda.compute` against the cu13 install"

`cccl-python`. `pip install -e python/cuda_cccl[test-cu13]` then
`ci/test_cuda_compute_python.sh`.

> "I added a new Numba CUDA cooperative primitive under `cuda.coop._experimental`.
> How do I wire up the tests?"

`cccl-python`. Explains the `cuda_coop` test pattern, points at
`ci/test_cuda_coop_python.sh`.

---

## 6. Performance

> "Write a CUB benchmark for the new `DeviceThreeWayPartition` algorithm using
> nvbench, with `%RANGE%` tuning annotations for items-per-thread"

`cccl-bench` (nvbench-template reference). Generates per-variant `.cu` files with
the shared `base.cuh` pattern.

> "Request a CI bench run for this PR — focus on device_reduce and device_scan,
> sm90 + sm120 GPUs only"

`cccl-bench` (ci-bench-request reference). Edits `ci/bench.yaml` with the filters,
appends `[bench-only]` to the commit message. Requires reset to template before merge.

> "Compare perf of this branch vs main for `thrust::sort` on 1M..256M element keys"

`cccl-bench` (local-run reference). Wraps `ci/bench/compare_git_refs.sh`.

> "Sweep CUB's `BlockScan` tuning space for sm120 and pick a new policy"

`cccl-bench` (tuning reference). Wraps the `cccl.bench` harness with
`CUB_ENABLE_TUNING=ON`, generates `.variant` targets, sweeps, picks the optimum.

> "Write a Python benchmark using `cuda.bench` for the new `cuda.compute.sort_pairs`
> binding"

`cccl-bench` + `cccl-python`. Python path uses `cuda.bench` with axis registration
and `bench.run_all_benchmarks(sys.argv)`.

---

## 7. Infrastructure & release

> "Bump the supported CUDA toolkit to 13.3"

`cccl-infra` (ctk-bump playbook). Edits `ci/matrix.yaml` (`ctk_versions`,
`devcontainer_version`, workflow rows), regenerates `.devcontainer/` via the
matrix-aware generator, verifies the workflow expansion. Refuses to hand-edit
individual `devcontainer.json` files.

> "Add support for gcc 15 to the host compiler matrix"

`cccl-infra` (compiler-bump playbook). Adds to `host_compilers`, cuda-specific
version table, workflow rows, regenerates devcontainers.

> "Cut a 3.2.0 release"

`cccl-infra` (release-cut playbook). Drives `ci/update_version.sh`, version files
per library (cub, thrust, libcudacxx, cudax), `cccl-version.json`,
`docs/VERSION.md`, Python package, workflows. Never hand-edits version files.

> "Add a new project under `c/parallel/` called `cccl-async` and wire it into CI"

`cccl-infra` (project-add playbook). `ci/matrix.yaml` workflow rows + `jobs:`,
`ci/project_files_and_dependencies.yaml` new key + deps, `CMakePresets.json`,
build/test scripts. Touches every infra file the project needs.

> "Pre-commit is failing — fix the formatting"

`cccl-precommit`. Runs the suite, reviews diffs, stages fixed files, re-runs.
Knows the auto-fix subset (clang-format, ruff, gersemi, end-of-file) vs the
non-auto-fix subset (codespell, mypy, shellcheck).

> "Build the docs locally"

`cccl-docs`. Runs `./docs/gen_docs.bash` (Linux-only, builds Doxygen 1.9.6 first
run, creates venv, runs Sphinx).

> "My new header isn't showing up in the API docs"

`cccl-docs` (doxygen-breathe-gotchas reference). Per-library Doxyfile inclusion
patterns, Breathe bridge config, custom `_ext/auto_api_generator.py`.

---

## 8. Decision-point prompts

> "I'm stuck — should I cherry-pick this fix onto `branch/3.1.x` or wait for the
> next 3.2 release?"

`cccl-clarify`. Three-step ladder: default reasoning from project conventions →
check the release cadence and the bug severity → ask the user with framed
options (cherry-pick / wait / hotfix release / break this down).

> "I have a clang-format diff but also a real code change in the same hunk —
> separate them?"

`cccl-commit` + `cccl-clarify`. Surfaces the choice as part of the interactive
chunk walkthrough.
