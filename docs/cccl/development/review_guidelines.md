# CCCL Review Guidelines

Guidelines for reviewing CCCL changes, distilled from past regressions. Each guideline
states what a human or AI reviewer should flag in a diff.

<!--
Many of the rules here were derived, with AI assistance, from actual PRs that introduced
regressions and the corresponding PRs that fixed them (listed as the rule's provenance).
Each derived rule was validated by having an AI reviewer, equipped with the candidate rule,
re-review the introducing PR's diff and confirm it flags the regression that shipped —
while a reviewer without the rule does not. See
https://github.com/NVIDIA/cccl/issues/11244 for more details.
-->

## Format

Each guideline is a section of the form:

```markdown
## <area>.<slug> (<severity>, <scope>)

<!-- provenance:
  #<introducing PR>→#<fixing PR> <short note>;
  review feedback on #<PR> (<link>) <short note>;
  ...
-->

<3–8 lines of imperative guidance: what to flag in a diff, and under which conditions it is acceptable>
```

- `<area>.<slug>` — stable rule id; `area` is one of `build`, `correctness`, `perf`,
  `api`, `abi`, `test`, `infra`, `docs`. New areas and slugs can be added as needed.
- `<severity>` — `critical` (must be addressed), `important` (not addressing requires a justification),
  or `suggestion` (worth considering, no action required).
- `<scope>` — which files/diffs the rule applies to.
- The provenance comment lists what the rule was distilled from: historical regressions
  (introducing PR → fixing PR) or review feedback that prevented a defect from shipping
  (link to the review comment); it is metadata for maintainers, not part of the rule.


## perf.tuning-refactor-verification (important, CUB tuning-policy selectors in `cub/device/dispatch/tuning/*.cuh` and perf-critical type/arch dispatch)

<!-- provenance:
  #3127→#3239 RLE tuning LOAD_LDG/LOAD_DEFAULT fallback swap;
  #3137→#3240 same bug in ReduceByKey tuning;
  #3138→#3236 dropped load_algorithm/store_algorithm fields SFINAEd sm90 to default;
  #1631→#1801 atomics backend refactor lost native-width codegen dispatch for scalar types;
  #6544→#7805 dispatch_streaming_arg_reduce_t stopped forwarding PolicyChainT to the new reduce::dispatch free function, silently defeating ArgMin/ArgMax tuning;
  #3125→#3174 deleting "unused on SM90" policy members broke multi-arch builds that still instantiate MaxPolicy for older-SM kernels (NVBug 5009941);
  the historical defects predate the constexpr policy_selector architecture, but the failure modes carry over
-->

A policy selector is a constexpr function from a compute capability and type information to a tuning
policy value. When a diff refactors one (or anything it calls, or the `*Policy` structs it returns)
without claiming a perf change, the function must return the same policy values for every input as
before — check equivalence per (architecture, type, size, operation) combination, not per code branch.
Typical silent breaks: a combination that used to hit tuned values now falls through to a fallback or
empty `optional` (no compile error, just an untuned policy); a field added/removed/reordered in a
policy struct shifting every positional brace-init (`ScanLookaheadPolicy{6, 104 - 1, 8, 2, 2}`); a
dispatch wrapper no longer forwarding its policy selector to an inner dispatch call. A claim of "no
SASS changes" verified on one test type is not sufficient; demand a SASS diff or benchmark sweep over
non-default/non-primitive value types and every affected architecture.

## correctness.shared-memory-overalignment (critical, `extern __shared__` storage or alignment of types placed in dynamic shared memory)

<!-- provenance:
  review feedback on #7868 (https://github.com/NVIDIA/cccl/pull/7868#discussion_r2880385130)
-->

Flag any `extern __shared__` storage (dynamic shared memory) with alignment greater than 16,
including indirectly via `alignas` on a type or data member placed in such storage. Before nvcc 13.1,
compiler bugs failed to retain alignment higher than 16 bytes in some cases, producing misaligned
accesses. Independently, a dynamic shared memory declaration with alignment > 16 increases the static
shared-memory padding for the entire translation unit, which can reduce occupancy of unrelated
kernels, including in downstream user code. CCCL code must not introduce any `extern __shared__`
storage with alignment above 16. A different design is required instead, e.g., manual alignment of a
byte buffer. Static shared memory with alignment > 16 is fine.
