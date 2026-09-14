# Compile-time benchmark CI contracts

The compile-time benchmark CI flow is configured from `ci/matrix.yaml` under
`compile_time.pull_request`.

## Matrix schema

Each config is a GitHub Actions matrix entry:

```yaml
compile_time:
  pull_request:
    - id: cccl-gcc13
      name: CCCL compile-time bench
      project: cccl
      gpu: rtx2080
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

Required config fields are `id`, `name`, `project`, `gpu`, `launch_args`,
`baseline_ref`, and `slices`. `project` is one of `cccl`, `pytorch`, `matx`, or
`rapids`. CCCL configs also require `preset` and `targets`;
RAPIDS configs require `targets`, which name the libraries to build. `args`,
and `artifact_retention_days` are optional.

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

In baseline comparisons, `threshold` is measured against the total selected
inclusive/exclusive impact across all matched traces. The per-side reports still
use `sort` for their own top-N ordering; comparison worse/better tables always
rank by total impact so a change repeated across many traces is not hidden by a
larger single-trace movement.

## Report contract

`summarize_events.py --slices <json>` writes per-slice CSVs under
`event_reports/<slice-id>/` and writes a normalized `event_reports/summary.json`
manifest. The manifest is the renderer contract; CSVs are human artifacts.
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
- Perfetto copies: `compile_time/perfetto_traces/current` and
  `compile_time/perfetto_traces/baseline`

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
