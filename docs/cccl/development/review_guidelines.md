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
- Rules are grouped by area, in the order `build`, `correctness`, `api`, `abi`, `perf`,
  `test`, `infra`, `docs`.

## build.compiler-matrix (important, all C++ code)

<!-- provenance:
  #534→#536 MSVC int128/asm;
  #1403→#1423 VLAs;
  #1320→#1417 NVHPC/GCC;
  ICC visibility →#1152;
  #1863→#1929 include cleanup broke non-NVRTC;
  #1915→#2341 host-system macro clobbered by mechanical typedef→using sweep;
  #5906→#5921 unguarded __is_trivially_copyable broke NVHPC/MSVC;
  #8957→#9003 atomicCAS in shared lit-test kernel broke tile-mode compilation;
  #10623→#10865 __uint128_t punning in __nv_atomic unavailable under tile mode/MSVC;
  #10927→#11014 plain if in an if-constexpr dispatch chain made 64/128-bit instantiations ill-formed under -mbmi2, invisible to CI;
  #6337→#6759 exception-macro refactor wrapped all of cuda_error.h in #if !NVRTC, removing __throw_cuda_error from NVRTC entirely;
  #3989→#4089 new usage site of the #ifdef-guarded ublkcp enumerator left unguarded, breaking NVHPC;
  #4871→#4891 compiler-crash workaround (_CCCL_NO_UNIQUE_ADDRESS removal) applied to the primary template but not the two-arg partial specialization;
  #4286→#8139 concept-constrained NTTP in ptx-json partial specializations rejected by NVRTC < 12.5
-->

Flag compiler-specific constructs unless guarded by a feature macro: `__int128`, GNU inline
`asm`, `__attribute__`, VLAs, visibility attributes, one-compiler warning suppressions, etc.
CCCL must build with GCC, Clang, MSVC, NVHPC/NVC++ (incl. `-stdpar`), and NVRTC; PR CI does
not cover all of these, so green CI is not sufficient. Compiler-detection refactors must
stay equivalent for every supported compiler — beware masquerading (Clang and NVHPC define
`__GNUC__`). Benchmarks and tests count too.

## build.fp16-implicit-ops (important, C++ code using `__half`/`__nv_bfloat16` or similar extended FP types)

<!-- provenance:
  #1735→#1785 static_cast<__half>/static_cast<float> instead of __float2half/__half2float in cub histogram ComputeScale
-->

Flag any use of the vendor-provided conversions or operators on CUDA extended floating-point types
(`__half`, `__half2`, `__nv_bfloat16`, `__nv_bfloat162`): `static_cast<__half>(f)`,
`static_cast<float>(h)`, arithmetic/comparison operators (`h1 + h2`, `h1 < h2`), and compound
assignments. The vendor headers compile these out when a user defines
`__CUDA_NO_HALF_CONVERSIONS__`/`__CUDA_NO_HALF_OPERATORS__` (or the HALF2/BFLOAT16/BFLOAT162
equivalents), so such code silently fails to build for those configurations. Use the explicit
intrinsics instead (`__float2half`/`__half2float`, `__hadd`/`__hgt`, …). `__nv_bfloat16`
arithmetic/comparison intrinsics require SM80 while CCCL supports sm75+: gate them with
`NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80, …)` and a float round-trip fallback via the explicit conversion
intrinsics (`__bfloat162float`/`__float2bfloat16_rn`; for `__nv_bfloat162`,
`__bfloat1622float2`/`__float22bfloat162_rn`). Candidate for a pre-commit grep.

## build.narrow-arithmetic-then-widen (important, C++/CUDA code computing a size/count/capacity/offset, including test files)

<!-- provenance:
  #7705→#9736 fixed_capacity_map tests computed capacity via static_cast<size_t>(num_keys * 2), multiplying in int before widening, breaking a clang-tidy CI check (pair auto-inferred as #9719→#9736)
-->

When a diff computes a size, count, capacity, or offset by adding or multiplying two operands and
widening the RESULT afterward — explicitly with a cast or implicitly through a wider
destination type, a wider function parameter, or return type — flag it as a
code smell: the operation executes in the narrower type and can silently overflow before the
result is widened. The author should either widen an operand before the operation
or, if the narrow result is intended, narrow the destination type so no widening occurs.

## correctness.pdl-restrict-aliasing (critical, CUDA kernels that call `_CCCL_PDL_GRID_DEPENDENCY_SYNC()` / `cudaGridDependencySynchronize()`)

<!-- provenance: manually added -->

Flag a kernel parameter marked both `const` and `_CCCL_RESTRICT` (or raw `__restrict`/`__restrict__`)
whose pointee is read after a `_CCCL_PDL_GRID_DEPENDENCY_SYNC()`/`cudaGridDependencySynchronize()`
call, when that memory is written by the preceding kernel. `const` + `restrict` together assert the
memory is read-only *and* uniquely accessed through this pointer, licensing the compiler to reorder
the load ahead of the sync. Acceptable only if a comment at the parameter explains why the memory can
never be concurrently aliased (e.g. no producer kernel writes it).

## correctness.pdl-cc-gate (important, CUB dispatch code in `cub/device/dispatch/*.cuh` and launcher factories launching kernels with programmatic dependent launch)

<!-- provenance:
  #9134→#9163 PDL enabled unconditionally (dependent_launch/use_pdl hardcoded to true) in five dispatch files, attempting PDL on PTX/SASS that never targeted sm_90+
-->

When a kernel is launched with PDL enabled (either directly or via a launcher factory like
`TripleChevronFactory` or `CudaDriverLauncherFactory`), flag any argument for
`dependent_launch`/`use_pdl` that does not consider the current device's compute capability queried
via `cub::detail::ptx_compute_cap`. PDL may only be enabled if
`cc >= ::cuda::compute_capability{9, 0}` (see `dispatch_find.cuh`).

## correctness.trivially-copyable-trait (important, generic code constraining or branching on trivial copyability)

<!-- provenance:
  #8210→#8254 warp_shuffle static_assert(cuda::std::is_trivially_copyable_v) rejected __half/__nv_bfloat16 and composites, breaking CUB consumers; reverted
-->

Flag `cuda::std::is_trivially_copyable(_v)` or `std::is_trivially_copyable(_v)` applied to a generic
value-type parameter;
use `cuda::is_trivially_copyable(_v)` instead, which supports more cases. The vendor headers give
`__half`/`__nv_bfloat16` non-trivial special members, so the standard trait reports false for them
(and aggregates of them) even though they are functionally copyable. Candidate for a pre-commit grep.

## api.internal-symbol-exposure (important, new implementation-detail types/functions)

<!-- provenance:
  #2591→#3209 CUB launcher factories and kernel-source getters added outside detail (pair auto-inferred from issue #2448)
-->

Flag any entity added to a public namespace which can be recognized as intended to be internal by its
spelling (e.g. snake_case in CUB, or prefixed with `__` in libcu++/cudax) or usage pattern (e.g. used
as utility for other functions, not documented, etc.). The entity should be marked internal as
appropriate.

## abi.missing-hide-from-abi (critical, inline functions in public headers whose behavior depends on the build configuration)

<!-- provenance:
  #2591→#3209 CUB kernel-source getters returned kernel pointers without _CCCL_HIDE_FROM_ABI;
  #5255→#5272 driver_api.h promoted into the public libcudacxx tree carrying plain-inline functions without _CCCL_HOST_API
-->

Flag an inline function in a public header whose result can differ between two copies of CCCL linked
into one binary (e.g., built against different CUDA toolkits or CCCL versions) — especially functions
returning kernel or function pointers — unless it is marked `_CCCL_HIDE_FROM_ABI` (or an attribute macro
that includes it, like `_CCCL_HOST_API`). With default visibility the linker keeps ONE definition
across all copies, so the losing copy's callers get the other build's result (e.g. a wrong kernel
pointer) — no build error, just wrong behavior at run time.

## correctness.cuda-driver-symbol-version-guard (critical, code calling CUDA Driver API symbols)

<!-- provenance:
  #2192→#5971 cudaGetDriverEntryPointByVersion gated only by a build-time CUDART_VERSION check, breaking when built against CTK>=12.5 but run against an older CUDA runtime (pair auto-inferred from issue #5970);
  #5976→#6895 (backport #6896) cuGetProcAddress switch reintroduced the same break by bootstrapping via the unversioned cudaGetDriverEntryPoint
-->

<!-- note:
  The Runtime API half of the historical incidents is no longer relevant to CCCL: cudart is statically
  linked everywhere (c/parallel pins CUDA_RUNTIME_LIBRARY STATIC; #7221 fixed the one shared-cudart
  mix), and the driver bootstrap in cuda/__driver/driver_api.h now dlopens libcuda directly instead of
  going through cudart.
-->

When a diff gates a call to a CUDA Driver API symbol introduced in a specific CUDA version behind a
build-time-only check (`_CCCL_CTK_AT_LEAST(...)`), flag it: `libcuda.so`/`nvcuda.dll` comes from the
installed display driver, which is independent of — and often older than — the CTK the binary was
built against, so the symbol can be absent at run time regardless of any build-time guard. Resolve
driver entry points through the versioned `cuGetProcAddress` bootstrap in
`cuda/__driver/driver_api.h` (which reports availability), or verify `cudaDriverGetVersion` before
the call. PR CI builds and runs with matched driver/CTK, so this only reproduces in the field.

## infra.pin-deps (important, CMake/CI/submodules)

<!-- provenance:
  #534 nvbench `#main` →#582
-->

Flag dependencies fetched by branch name (`CPMAddPackage("gh:org/repo#main")`, `GIT_TAG
main`); pin a commit or tag. Candidate for a pre-commit grep.

## correctness.stale-refs-after-rename (important, renames, moves, or splits of files, symbols, or modules anywhere in the repo)

<!-- provenance:
  #3177→#3192 cuda.parallel module split left docs automodule pointing at emptied package;
  #10012→#10042 docs flattening left stale path in a test comment and an empty api/thread toctree stub;
  #1075→#1108,#1110 lit.cfg path not updated after symlink removal broke local lit runs;
  #4537→#6516 NVTX macro rename left a stale #define in test_nvtx_disabled.cu, silently defanging a negative test;
  #4795→#4814 cudax detail→__detail rename swept ~120 files but missed the second example copy at top-level examples/cudax/
-->
<!-- note:
  A docs CI check failing on autodoc directives that yield no members (or on empty generated pages)
  would cover the last clause mechanically; retire it once such a check exists.
-->

When a diff renames, moves, or splits a file, macro, symbol, or module, `git grep` for the old name:
each remaining hit must be updated, or be classified as an unrelated entity that merely shares
the name. Doc directives (`automodule::`/`toctree::`) can go stale without containing the old name and
still build cleanly (autodoc renders emptied packages as blank pages) — verify the rendered docs, not
just the grep.

## correctness.workaround-breaks-constexpr (important, constexpr-marked)

<!-- provenance:
  #5939→#7059 `auto __tmp = mapping(); return __tmp.is_exhaustive();` workaround for a clang [[nodiscard]] warning made mdspan's is_exhaustive unusable in constexpr context on some compilers; propagated to is_unique/is_strided/stride in #6703
-->

When a diff introduces any workaround inside a `constexpr` function, verify the function is still
usable during constant evaluation on every supported compiler, not merely that it compiles as a
runtime call. The break only surfaces when a caller uses the function during constant evaluation,
which the unit tests may not exercise. Adding a test that evaluates the function at compile time is
recommended.

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

## perf.shared-primitive-consumers (important, shared thread/warp/block-scope primitives in cub/thrust/libcudacxx)

<!-- provenance:
  #2756→#2944 ThreadReduce SIMD/ILP rewrite regressed Select, ReduceByKey, and large-value-type Reduce;
  #4377→#6246 (backport #6265) ThreadReduce flattening refactor passed the raw Input type instead of the value type to the enable_*_v traits, permanently disabling all SIMD/ternary fast paths;
  #5507→#5542 vectorized-load selection demoted from compile-time to a runtime bool in the per-tile hot loop;
  #6099→#6452 (backport #6458) __bit_log2 swap in BlockRadixRankMatchEarlyCounts regressed radix sort up to 30%
-->

When a diff changes the implementation of a widely-reused low-level primitive (e.g. `ThreadReduce`,
warp/block scan/reduce helpers) in a way that can affect the generated SASS (a new fast path, a
refactor, a swapped helper function), flag it. Require a clean SASS diff for the entire PR, or
benchmarks across the primitive's consumers (not just its own microbenchmark). Cover non-standard
binary operators, large value types, and every architecture the change affects: a change that helps
one workload can silently regress a different algorithm layered on top of the primitive.

## perf.vector-init-before-overwrite (suggestion, `thrust::host_vector`/`thrust::device_vector` and c2h's `host_vector`/`device_vector` construction anywhere in the diff)

<!-- provenance: manually added -->

Flag a sized `thrust`/`c2h` `host_vector`/`device_vector` construction (`vector(n)`, `vector(n, value)`)
or `resize(n)` that doesn't use the `thrust::no_init` sentinel when the vector's content is never read
before being fully overwritten — as an algorithm's output, a kernel/copy destination, or an explicit
full assignment right after. The default overload value-initializes every element first, work the
overwrite immediately discards. Temporary-storage allocations are almost always a `no_init` candidate,
since callers assume that memory starts uninitialized. Does not apply when the vector is read before
being overwritten, or when the initial values are semantically required (e.g. an accumulator seed, or a
vector only partially written). When the element type is not trivially default-constructible,
or generic (e.g., in a template context), use `thrust::default_init` instead.

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
