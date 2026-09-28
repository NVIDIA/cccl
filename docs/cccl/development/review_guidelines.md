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
CCCL must build with GCC, Clang, MSVC, ICC, NVHPC/NVC++ (incl. `-stdpar`), and NVRTC; PR CI
does not cover all of these, so green CI is not sufficient. Compiler-detection refactors must
stay equivalent for every supported compiler: when an `#if`/`#elif` is rewritten from raw
macros (`__GNUC__`, `__clang__`, `..._HOST_COMPILER == ..._GCC`) to exact-identity macros,
re-evaluate it per compiler — Clang, ICC, and NVHPC masquerade by defining `__GNUC__`, so the
old branch matched them while the new one matches real GCC only. Tell-tale bug: a converted
allowlist (`#if GCC || CLANG || ICC`) gating an `#include` or feature that later code uses
unconditionally — NVHPC silently drops out; flag any converted allowlist missing an NVHPC
term (or not rewritten as a denylist) and any deleted `#else` fallback. Benchmarks and tests
count too — CCCL builds them with MSVC, which supports none of the above constructs.

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

## correctness.pdl-sync (critical, kernels launched with programmatic dependent launch)

<!-- provenance:
  #3114→#5456 (backports #5460, #5461) PDL grid-dependency sync in AgentMerge::consume_tile placed after the merge_partitions reads it was meant to guard, causing intermittent races/cudaErrorIllegalInstruction (issue #5297)
-->

When a diff enables programmatic dependent launch for a kernel by setting `dependent_launch` to true
at the kernel launcher, flag any global-memory access in the kernel's body that happens before a
call to `_CCCL_PDL_GRID_DEPENDENCY_SYNC` by that thread — the previous kernel's writes may not be
visible yet — unless the access has a comment explaining why a PDL sync can come later.

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
pointer) — no build error, just wrong behavior at run time. When a header is new to a shipped
include tree — including files moved or promoted from elsewhere (e.g. cudax or detail into
libcudacxx) — scan the whole file for bare `inline` function definitions, not just the changed
lines, and flag each one that lacks such a macro.

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

When a diff renames, moves, or splits a file, macro, symbol, or module, `git grep` the entire repo
(not just the directories the diff touches) for the old name: each remaining hit must be updated, or
be classified as an unrelated entity that merely shares the name. Beware CCCL's mirror trees: examples
live both under `<project>/examples/` and top-level `examples/<project>/`, so a bulk rename that
updates one copy can silently miss the other. Doc directives (`automodule::`/`toctree::`) can go stale without
containing the old name and still build cleanly (autodoc renders emptied packages as blank pages) —
verify the rendered docs, not just the grep.

## correctness.workaround-breaks-constexpr (important, constexpr-marked)

<!-- provenance:
  #5939→#7059 `auto __tmp = mapping(); return __tmp.is_exhaustive();` workaround for a clang [[nodiscard]] warning made mdspan's is_exhaustive unusable in constexpr context on some compilers; propagated to is_unique/is_strided/stride in #6703
-->

When a diff introduces any workaround inside a `constexpr` function, verify the function is still
usable during constant evaluation on every supported compiler, not merely that it compiles as a
runtime call. The break only surfaces when a caller uses the function during constant evaluation,
which the unit tests may not exercise. Adding a test that evaluates the function at compile time is
recommended.

## perf.partial-pdl (important, kernels launched with programmatic dependent launch)

<!-- provenance:
  #3114→#3199 PDL enabled at Partition/Merge triple_chevron launches but not the sibling BlockSort launch
-->

When a diff enables programmatic dependent launch for a kernel by setting `dependent_launch` to true
at the kernel launcher, open the full dispatch function (or equivalent) and enumerate EVERY kernel
launch it makes. All kernels should be launched using PDL, otherwise the performance gain is marginal.
Replacing calls to `cudaMemset` by kernels launched using PDL should be strongly considered and
pointed out as suggestions.

## api.type-replacement (critical, public types in thrust/libcudacxx/cub)
<!-- provenance:
  #262→#1249 (backport #1292) pair trivial copyability;
  #454→#1286,#1425,#1497 complex reverted three times;
  #6393→#6403 variant modularization dropped monostate include from the umbrella header
-->

When a diff reimplements, re-derives, or aliases any public type (`thrust::pair`/`tuple`/`complex`,
iterators, …) — e.g. replacing a hand-written struct with an alias of or derivation from a
`cuda::std::`/libc++ type — do not assume equivalence: open the replacement's definition and verify
every observable property of the old type is preserved: trivial copyability (any user-provided,
non-defaulted copy/move constructor or `operator=` in the replacement or its bases kills it) and
layout (downstream code `memcpy`s them and passes them as kernel arguments), size/alignment, implicit
conversions and promotions, overload resolution, and numerical behavior. Flag a lost property as a
present defect unless the diff adds `static_assert`s pinning it.

## build.windows-min-max-macro (important, C++ code calling `.max()`/`.min()` or naming a new member/trait `max`/`min`)
<!-- provenance:
  #8875→#9246 argument-annotation trait member named max, computed via unparenthesized numeric_limits<T>::max(), a preprocessor argument-count error under <windows.h>'s max/min macros;
  renamed to highest/lowest and parenthesized
-->

Flag an unparenthesized call to a function literally named `max`/`min` (e.g.
`std::numeric_limits<T>::max()`), and any new member or trait member named `max`/`min` (e.g.
`static constexpr T max = ...;` in a traits struct). On Windows, `<windows.h>` defines `max`/`min`
as function-like macros, breaking such code. Headers sandwiched between
`<cuda/std/__cccl/prologue.h>`/`epilogue.h` (libcudacxx, cudax) are safe for calls, but NOT for new
declarations named `max`/`min`: user code spelling `some_trait<T>::max` still expands the macro.
Everywhere else (CUB, Thrust, tests, examples), require the macro-safe spelling
`(std::numeric_limits<T>::max)()`. Always require a different member name (e.g. `highest`/`lowest`).
Candidate for a pre-commit grep.

## perf.benchmark-exec-tag-sync-without-sync-call (important, nvbench benchmark harness `state.exec(...)` calls)

<!-- provenance:
  #3114→#5350 merge_sort keys benchmark switched no_batch→sync while adding PDL although the exec lambda never synchronizes;
  reverted as an unnecessary workaround
-->

When a diff makes an nvbench `state.exec(...)` call use `nvbench::exec_tag::sync` (which tells
nvbench that the benchmark region will perform CUDA synchronization itself), or changes the lambda
body of a call already using such a tag, verify the lambda actually performs any explicit CUDA
synchronization (like `launch.get_stream().sync()`, `cudaStreamSynchronize`). Parallel algorithms in
Thrust and `cuda::std::` synchronize internally, except under `thrust::cuda::par_nosync`. Without a
sync, the measured time silently excludes some or all of the kernel's execution. If the lambda does
not sync, `exec_tag::no_batch` or `exec_tag::timer` is likely what was intended.

## api.narrowed-constraints-on-reimplementation (important, refactors/reimplementations of existing public APIs)

<!-- provenance:
  #1817→#2075 cub::DeviceMerge static_assert requiring identical value_type across both merge inputs, stricter than the thrust::merge implementation it replaced
-->

When a diff reimplements or reroutes an existing public API (port to a different backend, dispatch-layer
swap, internal rewrite), flag newly added `static_assert`/`enable_if`/concept/trait constraints that
reject inputs the previous implementation accepted (e.g. requiring identical `value_type` across two
input ranges where differing types previously worked). Rejecting previously accepted code is a breaking
change for downstream users; narrowing is only acceptable as a bug or conformance fix (the previously
accepted inputs produced wrong results or violated the documented contract), and must be called out in
the PR description.

## correctness.raii-move-no-disarm (critical, RAII/resource-owning/scope-guard types with move construction)

<!-- provenance:
  #5975→#10565 cudax scope_exit's move constructor was = default, copying the active flag without deactivating the moved-from source;
  both objects ran the cleanup action on destruction
-->

When a diff adds or defaults a move constructor for a type whose destructor conditionally runs an
action or releases a resource (an "active"/"engaged"/"owns" flag, a handle nulled on release), verify
the move disarms the moved-from source — resets its flag or nulls its handle, not merely copies it.
`= default` is a red flag: it member-wise copies the flag, so both objects fire the cleanup on
destruction. Require a test that moves the object and confirms the action fires exactly once and that the
moved-from object has been disarmed.

## correctness.verification-removed-without-replacement (important, any diff deleting a check or test)
<!-- provenance:
  #3743→#3866 dropping deprecated cub::Traits CATEGORY usage also deleted the static_assert cross-checks (old_IS_SMALL_UNSIGNED, "sanity check, remove eventually") comparing new type classification to the old one, breaking dispatch for library-extended types (NVBug 5121653);
  #3970→#9211 (issue #807) generate/raw_reference_cast simplification deleted the compile-fail harness (runtime_static_assert.h, unittest_static_assert.cu) with no replacement
-->

When a diff deletes a check verifying a property, but does not delete the checked entity, like a
`static_assert` cross-checking a new computation against an old one (tells: "sanity check" comments,
`old_*` names), a negative or compile-fail test (`*_fail*`, `*_static_assert*`, `UNSUPPORTED`/`XFAIL`
markers), a runtime assertion, then do not accept the deletion of the check, unless the diff shows the
property now holds by construction or adds a replacement check verifying the same property. Be
especially wary when the old side of a deleted cross-check reads a user-specializable trait or
customization point while the surviving new side uses a fixed standard trait: for user-extended types
the two can disagree, so dispatch silently changes with no compile error — that the old mechanism is
deprecated does not make the deletion safe.

## correctness.temp-storage-raw-alignment (critical, CUB/Thrust device-dispatch code allocating or indexing into `d_temp_storage`)

<!-- provenance:
  #6811→#9565 (backport #9781) warpspeed DeviceScan dispatch performed raw uint4 stores into d_temp_storage at a hand-computed offset, guarded only by a debug-only _CCCL_ASSERT and a "we probably need to ensure alignment" TODO, instead of routing through detail::alias_temporaries();
  misaligned callers hit Warp Misaligned Address faults on Blackwell (issue #9742)
-->

Flag any manual handling of the caller-supplied `d_temp_storage` in a dispatch: pointer arithmetic,
hand-computed offsets or alignment, or passing the raw pointer into a kernel argument. Only two forms
are allowed: (1) algorithms needing no temporary storage set `temp_storage_bytes` to 1 and never
touch the pointer; (2) algorithms needing one or more allocations must carve them out via
`detail::alias_temporaries` or `detail::temporary_storage::layout`, which alone are allowed to round
the base pointer up and report the required size.

## correctness.noexcept (critical, new or changed async/fallible public member functions)

<!-- provenance:
  #7705→#10888 fixed_capacity_map's *_async members were noexcept while __open_addressing_impl ("@throws cuda_error") used _CCCL_TRY_CUDA_API;
  the cooperative-group launch branches also had no error check at all, unlike their cg_size==1 siblings
-->

No exception may escape a `noexcept` function on any code path — an escaping exception calls
`std::terminate`, turning a recoverable error into a process crash, and it compiles cleanly with no
warning. Pay attention to throwing reached through helpers: `_CCCL_TRY_CUDA_API`, `_CCCL_THROW`, or
callees documented `@throws`.

## correctness.header-kernel-weak-linkage (critical, `__global__` kernels defined in headers)

<!-- provenance:
  #2641→#2656 templatized CUDASTF's callback_completion_kernel to dodge a multiple-definition linker error, risking runtime launch errors
-->

Flag a `__global__` function with an unused template parameter (`template <int = 0>`), or marked
`inline` — typically done to dodge a "multiple definition" linker error for a kernel defined in a
header. The linker collapses the weak host stubs to one, but each translation unit registers its own
fatbin, so a launch can resolve to a stub whose kernel was registered by a different TU and fail at
runtime. Hidden visibility (`_CCCL_KERNEL_ATTRIBUTES`) does not prevent this. Give the kernel internal
linkage instead: `static` or an unnamed namespace.

## build.no-long (important, C++/CUDA code, including tests)

<!-- provenance:
  #6068→#6081 c/parallel three_way_partition used `using OffsetT = long`, whose choose_signed_offset static_assert fails under MSVC (LLP64: long is 32-bit)
-->

Flag any use of `long`/`unsigned long` as a chosen type. `long` is 64-bit on LP64 Linux/macOS but
32-bit on LLP64 Windows (MSVC, clang-cl), so code assuming either width builds and passes on one
platform and silently truncates or fails on the other. Use a type that says what is meant:
`int32_t`/`uint32_t` or `int64_t`/`uint64_t` for exact widths, `size_t` for object sizes,
`ptrdiff_t` for pointer differences. Acceptable: `long` as a *supported* type for
traits, overload sets, type-list tests enumerating fundamental types, and external API signatures
that use it. Candidate for a pre-commit grep.

## correctness.ptx-asm-operand-index (critical, hand-written/generated inline PTX `asm volatile` blocks)

<!-- provenance:
  #3440→#8403 128-bit atomic CAS codegen template misindexed asm operands (mov.b128 read from output/undefined and cross-mixed compare/desired registers), returning success while writing garbled data (intro corrected from issue #8402)
-->

When a diff adds or edits an inline `asm volatile("..." : outputs : inputs : clobbers)` block
referencing operands by number (`%0`, `%1`, …), manually verify each `%N` against its declared position
(outputs first, then inputs, in constraint-list order) — the compiler only checks that `%N` is in
range, not that it refers to the intended operand. Watch for reads of output-only (`"="`) operands and
off-by-one indices after a reorder. Tests should read back the written values, not just a status.

## api.alias-template-ctad-gap (important, public type aliases wrapping a class template that supports CTAD)

<!-- provenance:
  #3686→#6093 host_mdspan/device_mdspan/managed_mdspan alias templates over cuda::std::mdspan with a substituted accessor; CTAD silently failed to compile (pair auto-inferred from issue #6076)
-->

When a diff introduces a type as an alias template, users cannot construct it via CTAD in C++17.
This is usually fine, unless the alias replaced a public entity that previously supported CTAD,
in which case the change breaks CTAD. Flag the alias template and require a test for CTAD to be added.

## correctness.offset-type-narrowing (important, CUB/Thrust device-algorithm dispatches and kernels templated on an offset type)

<!-- provenance:
  #2234→#4888 new cub::DeviceReduce RFA (deterministic reduce) kernel accepted a generic OffsetT but truncated it to int internally for grid sizing/indexing
-->

When a dispatch or kernel is templated on an offset type instead of using a fixed-width integral type,
the entire implementation behind it must support both 32-bit and 64-bit offsets. Flag any unguarded
conversion of an offset to a narrower type: `static_cast<int>(num_items)`, `int`-typed loop counters,
or grid-size arithmetic fed from `OffsetT`. Such code compiles and passes small tests but silently
misbehaves beyond 2^31 elements. A public API accepting a templated offset type must normalize it via
`cub::detail::choose_offset_t` immediately and pass the adjusted type to the dispatch and kernels, and
a unit test with a problem size larger than 2^32 is required. Not affected: implementations that accept
a generic offset type at the device-layer API but dispatch with a fixed 64-bit offset (e.g. lookahead
scan, transform).

## api.generic-param-replaces-concrete-type (important, public CUB/Thrust API overloads accepting a stream/config parameter)

<!-- provenance:
  #6204→#7915 (backport #8011) DeviceTransform Env=env<> default replaced the cudaStream_t stream parameter, breaking non-copyable types implicitly convertible to cudaStream_t
-->

When a diff replaces a concrete parameter type (e.g. `cudaStream_t stream`) with a generic templated
parameter (e.g. `Env env`), consider all possible call-site conventions the old parameter supported:
arguments could have been only implicitly convertible to the old type (e.g., a user stream wrapper
with `operator cudaStream_t()`), or non-copyable. A by-value template parameter binds to the
argument's own deduced type, not a potentially converted type from the old argument, so a non-copyable
stream-like lvalue that used to convert-then-copy now fails to compile. Either require an explicit
non-template overload taking the old type and constraining the generic overload, or turn the template
parameter into a const reference (e.g. `const Env& env`).

## correctness.element-index-to-byte-offset (important, CUDA kernels and dispatch code computing byte offsets from element indices)

<!-- provenance:
  #2086→#8803 (backports #8806–#8808) transform_kernel_ublkcp multiplied a 32-bit element offset by sizeof(T) in 32-bit arithmetic for bulk-copy addressing; for num_items*sizeof(T) > 4 GB the product wrapped and read wrong tile addresses, corrupting outputs (issue #8800)
-->

When an in-bounds element index or count stored in a 32-bit type is converted into a byte offset —
multiplied by `sizeof(T)`, used for pointer alignment, or applied to a `char*`/`std::byte*` cast of a
typed pointer — flag arithmetic performed in the 32-bit type: the element index may fit 32 bits while
the byte offset does not, so the multiplication must be widened to 64 bits first (e.g.
`offset * size_t{sizeof(T)}`).

## perf.jit-cache-key-vs-codegen-inputs (important, cuda.compute build-result caching)
<!-- provenance:
  #7657→#9596 (pair auto-inferred as #9475→#9596) histogram build cache keyed on runtime lower/upper level values and exact num_samples, forcing a recompile per distinct bounds (issue #9594)
-->

When a diff constructs or changes the cache key of a memoized build result (factories decorated with
`@cache_with_registered_key_functions`, the shared `cache_build_results` cache), audit every argument
in the key list individually — even in mechanical-looking refactors — and require the key to
consist of exactly the inputs that affect the generated code: dtypes, iterator kinds, operator
identity, compile-regime flags. Flag runtime kernel arguments in the key — scalar bounds such as
histogram lower/upper levels, exact element/sample counts like `num_samples` — since every distinct
runtime value then triggers a full recompile, silently destroying cache hit rates; canonicalize them
into their compile-relevant form first (dtype, 32/64-bit-offset flag). Scalars captured by a
JIT-compiled operator are deliberately keyed by value. Conversely, flag a key that omits a
compile-affecting input, which causes wrong-kernel reuse.

## test.sibling-config-drift (important, CMake test configuration)

<!-- provenance:
  #4802→#5242 _CCCL_HEADER_TEST added to the internal-headers test CMake config only, silently skipping the new prologue/epilogue check for public_headers and public_headers_host_only
-->

CCCL configures near-identical test targets from sibling CMake files, both within a project (the
libcudacxx header-test trio `libcudacxx/cmake/Libcudacxx*HeaderTesting.cmake`) and across projects
(`ThrustHeaderTesting.cmake`, `CubHeaderTesting.cmake`, `cudaxHeaderTesting.cmake`). When a diff adds
a compile definition, flag, or gating macro to one such file, check every sibling defining a
structurally similar target for the same addition — adding it to only one silently skips the check
for the others. Siblings are recognizable by near-identical file names and structure.

## perf.intrinsic-wrapper-codegen-parity (important, new generic wrappers over device intrinsics in libcudacxx/cub)

<!-- provenance:
  #3907→#10035 (pair auto-inferred as #8391→#10035) cuda::device::warp_shuffle memcpy-punned values through uninitialized locals and recomputed the predicate, inflating register pressure vs raw __shfl intrinsics
-->

When a diff introduces a generic (any-type) wrapper over a hardware intrinsic (warp shuffle/vote/match,
atomics), its generated code must be observable somewhere: either a codegen test comparing it against
the raw intrinsic for common types (e.g. a FileCheck test asserting the expected instruction and no
local-memory traffic, like the existing atomics/simd codegen tests), or the wrapper is used in kernels
covered by the benchmark SASS-diff CI job. Inefficiencies like `memcpy` through uninitialized locals
or recomputing outputs the instruction already produces have no functional symptom, so without one of
the two, unfavorable codegen in the new public API ships unseen.

## api.trait-specialization-member-shape (important, new specializations of standard-library customization-point traits)

<!-- provenance:
  #7439→#8486,#8488 thrust device_ptr/pointer/normal_iterator pointer_traits specializations defined rebind as a nested struct instead of an alias template;
  allocator_traits::rebind<U> silently named the struct, breaking RAPIDS via rmm's thrust_allocator
-->

When a diff specializes a primary template, where generic code names its members directly (traits like
`pointer_traits`, `allocator_traits`, `iterator_traits`), every member the specialization provides
must have the same shape as in the primary template: an alias template stays an alias template,
nested structs stay nested structs, constants stay constants with the exact same data type.
A wrong shape may still compile but carry a different meaning, staying invisible during compilation
while breaking generic consumers, often in third-party code.

## api.duplicated-derived-type-not-updated (important, CUB/Thrust algorithms with multiple public overload families layered over one dispatch)

<!-- provenance:
  #9289→#9676 DeviceReduce's env overloads recomputed accum_t inline instead of using the shared select_accum_t, missing the new no_init_t sentinel
-->

When a diff adds support for a new argument or case by extending a shared type-computation helper
(e.g. `select_accum_t`), check every other public overload family of the same algorithm for an
independently written computation of the same derived type — especially the `device_*.cuh` facade
over the touched `dispatch_*.cuh`, which is often not part of the diff; check it against the diff's
base revision. A copy that recomputes the type inline silently misses the new case. Prefer replacing
the inline computation with the shared helper.

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
without claiming a perf change, it must return the same policy values for every (architecture, type,
operation) combination as before. Typical silent breaks: a member-detection/SFINAE selector dropping
every architecture whose tunings lack a detected member (e.g. `load_algorithm`) to an untuned fallback
with no compile error; merged fallback paths whose hardcoded fields differ (`LOAD_DEFAULT` vs
`LOAD_LDG`) — reconstruct the old fallback from removed lines and diff it field by field; positional
brace-init shifts after policy-struct field changes; a dispatch wrapper no longer forwarding its policy
hub to an inner dispatch whose policy parameter is defaulted (an unused `PolicyChainT` in the wrapper
is a tell). Demand SASS-diff or benchmark evidence beyond one test type, on every affected architecture.

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

## infra.new-cuda-arch-rollout (important, diffs adding support for a CUDA architecture/SM number)

<!-- related: test.new-arch-coverage-gap covers the CI/test-exercise angle for arch-conditional branches in general; this rule is the specific checklist of sibling locations for rolling out a brand-new SM -->
<!-- provenance:
  #3550→#4931 sm_120 macros added to nv/target in January 2025; arch_traits<sm_120> (then living in
  cudax) wasn't added until June 2025, ~4 months later — code branching on NV_PROVIDES_SM_120 got wrong
  occupancy/shared-memory limits inherited from an older SM in the meantime
-->

When a diff adds support for a new CUDA architecture (SM number) not previously known to CCCL, go
through this checklist and verify every applicable entry was updated in the same diff (or a linked
follow-up PR):
- `libcudacxx/include/nv/detail/__target_macros` and `libcudacxx/include/nv/target` — the
  `NV_PROVIDES_SM_XXX`/`NV_IS_EXACTLY_SM_XXX` macro pair itself.
- `libcudacxx/include/cuda/std/__cccl/execution_space.h` — the new SM added to
  `_CCCL_KNOWN_CUDA_ARCH_LIST` (and to `_CCCL_KNOWN_CUDA_ARCH_SPECIFIC_LIST` if it needs a
  family-specific `arch_id`/`arch_traits` entry distinct from its base SM).
- `libcudacxx/include/cuda/__device/arch_traits.h` — new `arch_traits<arch_id::sm_XXX>()`
  specialization with its own limits, not copied from an older SM.
- `libcudacxx/test/libcudacxx/cuda/ccclrt/device/{all_arch_ids,arch_id,arch_id_fmt,arch_traits.c2h,is_specific_arch}`
  — exhaustive per-arch test tables.
- `cub/test/catch2_test_util_device.cu` — new `GEN_POLICY` entry in the `policy_hub_all` self-test.
- `cmake/CCCLCheckCudaArchitectures.cmake` — `all-major-cccl`/`all-cccl` resolution, if the new SM
  belongs in default multi-arch builds.
- Recommended: `ci/matrix.yaml` — new SM number added to at least one `sm:`/`codegen_target` job.

## infra.ci-flag-removal (important, `ci/*.sh`, `ci/matrix.yaml`, and other shared automation/config)
<!-- provenance:
  #493→#1458 removed -disable-benchmarks / ENABLE_CUB_BENCHMARKS env-var override when refactoring ci/build_cub.sh;
  #7919→#10057 (via prerequisite #8160) a PR titled "Remove CuPy upper bound" also silently dropped 12.0 from ctk: lists in ci/matrix.yaml, cutting CTK 12.0 python CI coverage unnoticed for months (issue #8156);
  #4924→#5543 release-wheels.yml simplification dropped the -p "*${comp}*" filter from gh run download, breaking wheel releases (pair auto-inferred as #5541→#5543)
-->

When a diff changes CI infrastructure — `ci/matrix.yaml`, CI shell scripts, build/test scripts,
workflow files — and removes or restricts CI coverage in any way (a version dropped from a job row's
value list, a CLI flag or `${VAR:=default}` override deleted, a job or filter removed), cross-check
the PR title and description: the removal must be intended and clearly pointed out. When such a script
is refactored, enumerate every CLI flag and env-var override on the deleted lines and verify each is
still honored afterward — a removed `${VAR:=default}` replaced by an unconditional `VAR=...` looks like
a harmless rewrite but silently ignores callers who set the knob. If a removal looks accidental — e.g.
the PR's stated purpose is unrelated — flag it and have the author confirm the removal is intended.
A coverage drop produces no CI failure and can go undetected for months.

## docs.link-resolves (important, diffs adding or changing hyperlinks in docs, comments, or messages)
<!-- provenance:
  #10887→#10895 bulk CUDA-guide link migration pointed memcpy_async performance guidance at the device-callable-APIs appendix instead of the async-copies page
-->

When a diff adds or changes a hyperlink, verify the URL resolves, including the `#fragment`. In a
bulk link migration (many `-`/`+` URL pairs), do NOT accept "the new anchor exists" as proof: for
each changed link, mechanically compare the OLD fragment's words against the NEW path+fragment. If
the old anchor named a specific topic (e.g. `#performance-guidance-for-<X>`, `#using-<X>`) but the
new link drops those qualifier words and points at a bare `#<X>` anchor on an
`appendices`/`device-callable-apis`/reference page, flag it — the link was downgraded from topical
guidance to an API listing. Likewise flag when surrounding prose promises performance or how-to
content but the new target is a reference/appendix page; the correct target is usually a
special-topics page — sibling links in the same diff pointing there for the same subject are a
strong hint.
