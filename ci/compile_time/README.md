# Compile-time benchmark CI contracts

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
inclusive/exclusive impact across all matched traces. With `--stability-filter`,
it applies to the diagnostic impact remaining after build drift correction.
The per-side reports still
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

With `--stability-filter`, comments instead show the raw matched-corpus total,
the top five diagnostic regression candidates across slices, the top three
improvements in a collapsed section, and up to three common headers separately.
Up to three localized/sparse changes appear in a separate collapsed section
without an established consistency claim, separately from consistent diagnostics.
Warnings remain visible. The complete `comparison/all.csv` for every slice
retains all comparable keys, including zero changes and filtered rows, without
threshold or top-N censoring. Existing per-side top-N CSVs remain available.

## TU consistency policy

The comparison wrapper enables `--stability-filter`. All report results come
from the initial baseline/current builds; there is no secondary compilation or
random resampling. Direct summarizer invocations keep aggregate-only reporting
unless this flag is used.

Occurrences and trace fragments are summed within each matched generated TU.
The raw paired changes are tested for directional consistency: the null is that
at most 75% of contributing TU contexts move in the given direction. Exact
one-sided binomial tails are computed for both directions, counting zero
changes against either direction. Holm adjustment covers both directions of
**all** comparable diagnostic keys across **all** slices, before seconds
thresholds or top-N selection. A row passes at adjusted p <= 0.05.

This is a model-based test of cross-context consistency, conditional on
independent TU directions with a common probability. It is not a causal test of
a revision effect or a confidence interval for the entire build. Shared runner
bias cannot be estimated from one build pair. Header-test inputs also overlap,
so independence is an assumption, not something guaranteed by matching traces.
See [STATISTICS.md](STATISTICS.md) for the derivation, justification, and limits.

For diagnostic ranking only, the median ratio of positive current/baseline TU
total durations supplies a build drift factor `g` (at least three valid TUs are
required). Diagnostic impact is `sum(C_i - g * B_i)`. The whole-build headline
and raw header impacts remain unadjusted. Inferential p-values use raw changes,
so the estimated drift factor is not treated as a known nuisance parameter.
A consistent diagnostic also needs raw and adjusted aggregate directions to
agree and adjusted impact to exceed its slice's seconds threshold.

Per-TU residuals divided by baseline occurrence count supply descriptive
median/MAD columns. Counts weight the total impact, without creating extra
trials. This preserves increased inclusion costs even when the per-occurrence
average is unchanged. Average change times count alone is not a noise correction.

Sparse and localized changes remain available as inspection candidates with
their p-values, separately from consistent rankings. All comparable keys,
including zeros and failed tests, remain in `comparison/all.csv`. Common headers
(present in at least 80% of matched TUs) have a separate presentation section;
no header names are blacklisted. The manifest records the full hypothesis
family size and test policy.

## Comparison build design

The wrapper creates fresh sibling current/baseline worktrees under the same
build directory/filesystem and uses identical instrumentation. The current
snapshot includes tracked working-tree changes and non-ignored new files
outside `build/`. Both sides are configured before measurement, and each
warms the same instrumented object shared by both configurations before cleaning object outputs and warm-up
trace files. This adds two object compilations, not two full builds.

`-build-order auto` alternates order using the parity of GitHub run number plus
attempt, or uses baseline-first outside CI. `-build-order current-first` and
`-build-order baseline-first` allow reproduction. The workflow explicitly
forwards the run number/attempt into the container. `build_pair.json` records
commit identities, order, and warm-up policy; its content is also embedded in
the summary manifest. Compiler launcher cache settings and their environment
initializers are disabled in both measured builds.

Comparison mode requires fresh configure/build; `-skip-configure` and
`-skip-build` cannot be combined with `-baseline-ref`. Replay archived traces
with `summarize_events.py` instead. On build failure, partial measured traces
are preserved and an earlier successful summary is not reused. Temporary
worktrees are cleaned without changing the caller's source files.

The reusable workflow uses the sticky-comment header
`compile-time-bench-<config-id>` with `hide_and_recreate: true`, so previous
comments for the same config are archived as outdated.
