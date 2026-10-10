# Compile-time benchmark CI contracts

The compile-time benchmark CI flow is configured from `ci/matrix.yaml` under
`compile_time.pull_request`. CCCL runs on PRs. Third-party configurations run
weekly and can be enabled on a PR by adding
`[run-third-party-compile-time-bench]` to its latest commit message or placing
configurations in `compile_time.override`.

## Matrix schema

Each config is a GitHub Actions matrix entry:

```yaml
compile_time:
  pull_request:
    - id: cccl-gcc13
      name: CCCL compile-time bench
      project: cccl
      runner: linux-amd64-cpu32
      launch_args: "--cuda 13.3 --host gcc13"
      baseline_ref: origin/main
      preset: all-dev
      targets:
        - cub.headers.base
      args: "-arch 75"
      slices:
        - id: total-compilation
          title: TU total compilation
          filter: total-compilation
          timing: inclusive
          sort: total
          top: 15
          threshold: 0.001
```

Required config fields are `id`, `name`, `project`, `runner`, `launch_args`,
`baseline_ref`, and `slices`. `project` is one of `cccl`, `pytorch`, `matx`, or
`rapids`. CCCL configs also require `preset` and `targets`;
RAPIDS configs require `targets`, which name the libraries to build. `args`,
and `artifact_retention_days` are optional.

`runner` is the full GitHub Actions runner label. Compile-time jobs use
`linux-amd64-cpu32` runners with 500 GB of disk space. These jobs only build;
CCCL configs specify an explicit CUDA architecture instead of `native` so they
can run without a GPU.

Required slice fields are `id`, `title`, `filter`, `timing`, `sort`, `top`, and
`threshold`. Optional `group_by: primary-template` aggregates template
instantiation specializations under NVCC's reported primary-template name; it
is valid with the built-in template-instantiation filters. Slice `children` may
be used to group nested report sections in the PR comment. Empty slice sections
are omitted recursively by the renderer unless the summary manifest carries
warnings for that slice.

The third-party CI configurations include both concrete template-instantiation
results and a `primary-template` grouped slice. The CCCL public-target
configuration omits the grouped slice because downstream instantiation patterns
are not that benchmark's subject.

`ci/compile_time/parse_matrix.py ci/matrix.yaml --workflow pull_request` emits
the GitHub Actions matrix JSON. Missing or empty `compile_time.pull_request`
emits `{"include":[]}`.

The weekly workflow selects the third-party entries from this matrix and runs
each against the head commit of the preceding scheduled weekly run. Each
comparison builds its baseline in the current runner environment; it does not
reuse old trace artifacts. The weekly workflow uploads reports and traces
without posting a PR comment.

The PR workflow uses a non-empty `compile_time.override` list in place of the
normal compile-time matrix and enables its third-party entries without an
opt-in tag. Override entries use the same schema as `compile_time.pull_request`.
Third-party entries compare against the preceding weekly run. Reset the
override list to empty before merging.

With an empty override, the PR workflow selects only CCCL entries unless the
opt-in tag is present. With the tag, it also runs the third-party comparisons.
Both routes include results in the combined PR comment. Existing project skip tags and
`[skip-compile-time-bench]` still take precedence.

In baseline comparisons, `threshold` is measured against the total selected
inclusive/exclusive impact across all matched traces. The per-side reports still
use `sort` for their own top-N ordering; comparison worse/better tables always
rank by total impact so a change repeated across many traces is not hidden by a
larger single-trace movement.

The comment retains its regression/improvement tables and full event names.
Event identities present on only one side appear in those same tables with
zero time on the absent side and zero matched traces. Matched and unmatched
rows share the configured top-N total-impact ranking.

Percent changes, mean deltas per trace, event counts, device-pass means, and
compilation totals remain available in the CSVs and summary manifest. These
measurements come from one build per revision; repeated measurements are needed
to establish significance. Source compilation rows aggregate build objects
using that source. Compilation totals sum per-object elapsed time, rather than
whole-build elapsed time.

Added and removed occurrences are reported separately when an event identity
appears on only one side of a trace pair, including unpaired trace files. They
use the same absolute threshold and total-cost ranking. A key can be matched in
some traces and added or removed in others. Inclusive timings and occurrence
costs can overlap across rows and tables; do not add them together. Matched
exclusive comparisons retain their existing comparable-child subtraction.

Third-party builds pass `--cccl-only`. File processing selects only the CCCL
checkouts used by MatX/RAPIDS and the custom CCCL installation in PyTorch's
copied CUDA tree. Checkout and installation prefixes are mapped to paths from
the CCCL repository root; consumer, other dependency, compiler, and SDK headers
are excluded. Symbol slices select the owning CCCL namespace, rather than
matching CCCL names in a consumer function's argument or return types.
Device-pass means are derived from NVCC's per-thread architecture metadata;
events without that metadata remain included in the aggregate timings.

For a focused PR run, keep only MatX in `compile_time.override` and use
`[bench-only] [skip-sass-diff]` in the commit message. `[bench-only]` skips the
ordinary matrix, devcontainer, docs, and third-party smoke builds while keeping
compile-time benchmarks enabled. Explicit third-party skip tags still apply.

## Report contract

`summarize_events.py --slices <json>` writes per-slice CSVs under
`event_reports/<slice-id>/` and writes a normalized `event_reports/summary.json`
manifest. The manifest is the renderer contract; CSVs are human artifacts.
The manifest starts with `status: incomplete`, checkpoints each completed
slice, and ends with `status: complete`. A failed run records `status: failed`,
the error, and any completed slices so the CI comment can show partial results
without presenting them as a successful comparison.
Trace JSON is parsed once per file and every slice is applied in that pass.
`--jobs N` parses files in parallel (default `min(cpu count, 8)`; `--jobs 1`
stays in-process).
Configured slices that match no events, have no matching trace files, or have no
comparable event keys record warnings in the manifest so reporting failures are
not presented as ordinary no-regression results.

For a single grouped template report, use:

```bash
ci/compile_time/summarize_events.py <trace-dir> \
  -f template-instantiation -i --sort total -n 25 \
  --group-by primary-template
```

The equivalent slice setting is `"group_by": "primary-template"`. Grouped
reports sum the selected timings and event counts of all specializations with
the same NVCC-reported primary-template label. Baseline comparisons apply the
same grouping on both sides.

In comparison mode, the wrapper preserves:

- current raw traces: `compile_time/raw_traces`
- baseline raw traces: `compile_time/baseline_raw_traces`
- when enabled, Perfetto copies: `compile_time/perfetto_traces/current` and
  `compile_time/perfetto_traces/baseline`

Third-party CI entries pass `-no-prepare-perfetto` to avoid storing another
copy of each large trace. Their raw current and baseline traces remain available
as artifacts and can be processed with `prepare_traces.py` after download. The
wrapper still prepares Perfetto copies by default for interactive runs.

MatX adds time tracing through `nvcc_time_trace.py`, a CUDA compiler launcher.
The launcher passes the flag directly to NVCC and uses one internal compiler
thread. CUDA 13.3 can fail to open temporary input files when time tracing is
combined with parallel device-image compilation. Translation units still build
in parallel using the configured build job count.

PyTorch uses the CUDA 13.4/GCC 14 environment defined in
`pytorch-devcontainer.json`: CUDA 13.3 PTXAS fails when reading its largest
profiling traces. Its CI entry passes `-summary-jobs 1` because individual
traces can exceed 20 GB. This option controls reporting workers independently
of build parallelism and applies the configured slices to each trace pair.
PyTorch profiling builds target only SM80 (`TORCH_CUDA_ARCH_LIST=8.0`) for both
CCCL revisions. PyTorch adds no per-source architectures for this target; the
multiarchitecture RowwiseScaledMM build exceeded six hours before the baseline
could start. All extracted CUDA translation units are still built.

CI uploads reports and raw traces from their build locations, avoiding another
local copy before artifact upload. RAPIDS trace collection assigns nested
projects, such as `cudf_kafka`, to their own labels instead of duplicating
their traces under a parent project.

## PR comments

`render_pr_comment.py` reads `summary.json`, config metadata, and an artifacts
URL. Its `--fragment` mode wraps one configuration in a `<details>` section for
the CI report. Regressions and improvements are rendered in separate
`<details>` blocks and are never mixed in one table. Warnings are rendered
separately and keep their slice visible even when there are no
regression/improvement rows.

Each reusable workflow run uploads its configuration fragment. The parent PR
workflow passes those artifacts to `combine_pr_comments.py` in matrix order and
posts one sticky comment containing all enabled configurations. The shared
`compile-time-bench` header uses `hide_and_recreate: true`, so the previous
combined comment is archived as outdated. If the detailed fragments would
exceed the comment body budget, the workflow posts each configuration's result
counts and links to the full fragment artifacts instead.
