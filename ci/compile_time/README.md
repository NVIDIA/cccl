# Compile-time benchmark CI contracts

## Third-party trace history and single-dataset reports

`compile_time.third_party` in `ci/matrix.yaml` selects the MatX, PyTorch, and
RAPIDS builds collected by the weekly workflow. Each configuration builds the
current CCCL revision without resolving or rebuilding a baseline. RAPIDS retains
both its tests-disabled and tests-enabled build phases; unchanged traces carried
over between phases are collected only once.

The standalone **Third-party compile-time trace collection** workflow can be
dispatched for one configuration or all configurations. PRs opt in with
`[run-third-party-compile-time-bench]`, or temporarily select configurations in
`compile_time.third_party_override`. The override takes precedence over the tag
and must be emptied before merging. Existing project skip tags and
`[skip-compile-time-bench]` suppress the corresponding trace builds. These jobs
are informational and do not change the existing CI merge gate.

For a PR run limited to third-party trace collection and reporting, use
`[run-third-party-compile-time-bench] [bench-only] [skip-sass-diff] [skip-cccl-compile-time-bench]`
in the commit message. The last tag skips only the existing CCCL public-header
comparison; `[skip-compile-time-bench]` suppresses all compile-time jobs.

### Original datasets

The `compile-time-<config-id>-traces` artifact contains:

- `raw_traces/`: **unmodified** object-adjacent NVCC JSON traces, copied rather
  than rewritten or hard-linked to the build tree.
- `dataset.json`: CCCL and consumer commit IDs, original path root, compiler
  versions and CMake flags/architectures, workflow/run identity, build and
  collection outcomes, and trace count.
- `config.json` and `slices.json`: the build configuration and analysis slices.

The workflow uploads this artifact **before analysis**. Failed builds still
collect any partial traces, and reporting failures cannot remove the uploaded
dataset. Each run/attempt uses a separate build directory; local collection also
refuses to overwrite an existing dataset. An empty collection fails the job.
Forced runner termination can prevent cleanup/upload from completing.

Trace, report, and comment artifacts request **90 days** of retention, GitHub's
maximum for public repositories. Lower repository, organization, or enterprise
limits still apply. This does not extend old artifacts or retain data beyond
GitHub's limits; deleting a workflow run also deletes its artifacts. Download
datasets before their `expires_at` time if they must survive longer:

```bash
gh api repos/NVIDIA/cccl/actions/runs/RUN_ID/artifacts \
  --jq '.artifacts[] | {name, created_at, expires_at}'
gh run download RUN_ID --repo NVIDIA/cccl \
  --name compile-time-matx-gcc14-traces --dir dataset
```

### Analysis and reports

Each dataset produces up to 50 entries, ranked by total time, for TU total
compilation, direct file processing, function-body parsing, template
instantiation, and instantiation grouped by primary template. Header reports
retain CCCL headers only and display paths relative to the CCCL checkout,
including installed headers in PyTorch's copied CUDA tree. Symbol reports select
CCCL namespace owners. TU totals describe the whole consumer compilation.
The parser continues to parse each trace once and apply all slices, using
multiple workers (one for large PyTorch traces to bound memory).

Tables appear in workflow job summaries and, for PR runs, report comments.
Comments are paginated without dropping ranked items. An individual event name
too large for a comment links to its complete name in the CSV/JSON artifact;
archived traces and full report names remain unchanged. This path has no
baseline, regression, or improvement tables.

Collect locally inside the matching devcontainer, without analysis:

```bash
CCCL_BUILD_INFIX=trace-history ./ci/build_third_party_compile_time.sh -project matx
```

Analyze a downloaded dataset without building anything:

```bash
python3 ci/compile_time/summarize_events.py dataset/raw_traces \
  --repo-root "$(jq -r .repo_root dataset/dataset.json)" \
  --cccl-only --slices dataset/slices.json --jobs 8 -o dataset/event_reports
python3 -m ci.compile_time.render_trace_report \
  --dataset dataset --output dataset/comments \
  --artifacts-url https://github.com/NVIDIA/cccl/actions/runs/RUN_ID#artifacts
```

Use `--jobs 1` for PyTorch. The recorded path root lets analysis normalize the
original compiler paths without recreating the historical checkout.

## Existing CCCL comparison jobs

The compile-time benchmark CI flow is configured from `ci/matrix.yaml` under
`compile_time.pull_request`.

## Matrix schema

Each config is a GitHub Actions matrix entry:

```yaml
compile_time:
  pull_request:
    - id: public-headers-gcc13
      name: Public headers compile-time bench
      gpu: t4
      launch_args: "--cuda 13.3 --host gcc13"
      baseline_ref: origin/main
      preset: all-dev
      targets:
        - cub.headers.base
      args: "-arch native"
      slices:
        - id: total-compilation
          title: TU total compilation
          filter: total-compilation
          timing: inclusive
          sort: total
          top: 15
          threshold: 0.001
```

Required config fields are `id`, `name`, `gpu`, `launch_args`,
`baseline_ref`, `preset`, `targets`, and `slices`. `args`, `comment`, and
`artifact_retention_days` are optional.

Required slice fields are `id`, `title`, `filter`, `timing`, `sort`, `top`, and
`threshold`. Slice `children` may be used to group nested report sections in the
PR comment. Empty slice sections are omitted recursively by the renderer unless
the summary manifest carries warnings for that slice.

`ci/compile_time/parse_matrix.py ci/matrix.yaml --workflow pull_request` emits
the GitHub Actions matrix JSON. Missing or empty `compile_time.pull_request`
emits `{"include":[]}`.

In baseline comparisons, `threshold` is measured against the total selected
inclusive/exclusive impact across all matched traces. The per-side reports still
use `sort` for their own top-N ordering; comparison worse/better tables always
rank by total impact so a change repeated across many traces is not hidden by a
larger single-trace movement.

## Report contract

`summarize_events.py --slices <json>` writes per-slice CSVs under
`event_reports/<slice-id>/` and writes a normalized `event_reports/summary.json`
manifest. The manifest is the renderer contract; CSVs are human artifacts.
Trace JSON is parsed once per file and every slice is applied in that pass.
`--jobs N` parses files in parallel (default `min(cpu count, 8)`; `--jobs 1`
stays in-process).
Configured slices that match no events, have no matching trace files, or have no
comparable event keys record warnings in the manifest so reporting failures are
not presented as ordinary no-regression results.

In comparison mode, the wrapper preserves:

- current raw traces: `compile_time/raw_traces`
- baseline raw traces: `compile_time/baseline_raw_traces`
- Perfetto copies: `compile_time/perfetto_traces/current` and
  `compile_time/perfetto_traces/baseline`

## PR comments

`render_pr_comment.py` reads `summary.json`, config metadata, and an artifacts
URL, then writes the sticky PR comment body. Regressions and improvements are
rendered in separate `<details>` blocks and are never mixed in one table.
Warnings are rendered separately and keep their slice visible even when there
are no regression/improvement rows.

The reusable workflow uses the sticky-comment header
`compile-time-bench-<config-id>` with `hide_and_recreate: true`, so previous
comments for the same config are archived as outdated.
