---
name: cccl-ci-benchmarks
description: "Request CCCL benchmark runs in PR CI by editing `ci/bench.yaml`, or launch benchmark workflows directly via `gh workflow run`. Walks the user through filter selection (CUB ninja-target regex / Python path regex), GPU selection, and the `[bench-only]` commit-tag convention. Use when the user wants to benchmark a change on PR CI, or trigger a one-off benchmark workflow. Trigger phrases: \"benchmark this PR\", \"request a perf run\", \"compare benchmarks before/after\"."
---

# cccl-ci-benchmarks

Two routes: PR-driven (edit `ci/bench.yaml`, push) and direct dispatch (`gh workflow run`).

`ci/bench.yaml` holds the request; `ci/bench.template.yaml` is the empty template CI checks against. Both must
match to merge.

## Route 1 — PR-driven

1. **Edit `ci/bench.yaml`:**
   - Add CUB benchmark regexes under `benchmarks.filters.cub` (matched against ninja target names, e.g.
     `^cub\.bench\.for_each\.base`).
   - Add Python benchmark path regexes under `benchmarks.filters.python` (matched against paths under
     `benchmarks/`, e.g. `compute/reduce/sum\.py`).
   - Uncomment at least one GPU under `benchmarks.gpus`: `t4`, `rtx2080`, `rtxa6000`, `l4`, `rtx4090`, `h100`,
     `rtxpro6000`. Pools are shared — pick conservatively.
   - Optionally adjust `launch_args` (e.g. `"--cuda 13.2 --host gcc14"`).

2. **Append `[bench-only]`** to the commit message — skips non-benchmark CI (equivalent to
   `[skip-matrix][skip-vdc][skip-docs][skip-tpt]`).

3. **Push.** Inspect dispatched jobs via `gh run view <RUN_ID>`.

4. **Reset before final merge.** Restore `ci/bench.yaml` to match `ci/bench.template.yaml` (empty filters, no GPUs
   uncommented).

## Route 2 — direct dispatch

If a benchmark workflow exists for direct dispatch (`gh workflow list --repo NVIDIA/cccl`):

```
gh workflow run <workflow-name>.yml --repo NVIDIA/cccl --ref <branch> -f <input>=<value>
```

Return the run URL. `gh workflow run` is mutating; prompts every use.

## Defaults

From `ci/bench.yaml`'s `Advanced` block:

- `base_ref: "origin/main"` — what to compare against.
- `test_ref: "HEAD"` — what to test.
- `arch: "native"` — usually fine; can be a list like `"80;90"`.
- `nvbench_args` — preset with timeout / skip-time / stopping criterion / throttle handling.

## Pitfalls

- Forgetting to uncomment a GPU → no jobs run.
- Forgetting `[bench-only]` → wasteful full-CI run alongside.
- Not resetting `ci/bench.yaml` before merge → merge blocked.
