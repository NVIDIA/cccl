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
iterators, …), verify every observable property of the old type is preserved: trivial copyability and
layout (downstream code `memcpy`s them), size/alignment, implicit conversions and promotions, overload
resolution, and numerical behavior.

## build.windows-min-max-macro (important, C++ code calling `.max()`/`.min()` or naming a new member/trait `max`/`min`)

<!-- provenance:
  #8875→#9246 argument-annotation trait member named max, computed via unparenthesized numeric_limits<T>::max(), a preprocessor argument-count error under <windows.h>'s max/min macros;
  renamed to highest/lowest and parenthesized
-->

Flag an unparenthesized call to a function literally named `max`/`min` (e.g.
`std::numeric_limits<T>::max()`), and any new member or trait named `max`/`min`. On Windows,
`<windows.h>` defines `max`/`min` as function-like macros, breaking such code. Headers sandwiched
between `<cuda/std/__cccl/prologue.h>`/`epilogue.h` (libcudacxx, cudax) are safe; everywhere else
(CUB, Thrust, tests, examples), require the macro-safe spelling `(std::numeric_limits<T>::max)()`
and prefer other member names. Candidate for a pre-commit grep.

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
property now holds by construction or adds a replacement check verifying the same property.

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

## docs.link-resolves (important, diffs adding or changing hyperlinks in docs, comments, or messages)

<!-- provenance:
  #10887→#10895 bulk CUDA-guide link migration pointed memcpy_async performance guidance at the device-callable-APIs appendix instead of the async-copies page
-->

When a diff adds or changes a hyperlink, verify the URL actually resolves, including the `#fragment`:
the anchor must exist on the target page. CI runs no link checker, so a broken or misdirected link
ships silently. Also check that the target page covers the topic the surrounding prose promises.
## build.deprecation-default-mismatch (important, CMake/config changes)

<!-- provenance:
  #2166→#2217 added C++14 deprecation warning without bumping CUB/Thrust's own default CMake dialect off C++14
-->

<!-- note:
  Incident summary: #2166 deprecated building with C++14 while THRUST_CPP_DIALECT still defaulted to 14,
  so every default-configured standalone build was immediately flooded by the new warning; #2217 bumped
  the default nine days later. CI did not catch it because CI pins every configuration value (presets/
  matrix pass the dialect explicitly; the C++14 jobs set *_IGNORE_DEPRECATED_* suppressions), so CMake
  defaults are a structural CI blind spot — only out-of-CI users exercise them.
  Triage: possibly too seldom to keep as its own rule (single incident); if kept, consider reframing
  around the broader mechanism "configuration defaults are never tested by CI".
-->

When a diff adds a deprecation warning/error tied to a specific configuration value (C++ dialect,
compiler version, a flag's value), check whether the project's own default configuration (CMake option
defaults, presets, CI scripts) still resolves to that now-deprecated value. If so, every user building
with un-overridden defaults is flooded by the new warning immediately; the default must be bumped in
the same change that adds the warning.

## build.packaging-file-paths (important, setup.py/pyproject.toml/MANIFEST.in in `python/*`)

<!-- provenance:
  #2211→#4212 (backported as #4217, #4218) license_files=('../../LICENSE',) escaped the package directory and broke under a newer setuptools
-->

<!-- note: was a single incident with the license file, seems not worth it -->

Flag packaging metadata (`license_files=`, `data_files=`, `package_data=`, `[tool.setuptools]` file
lists, `MANIFEST.in` includes) referencing a path outside the package's own directory via `../`.
Packaging tools only reliably include files inside the package/sdist root; an escaping path can work
on one setuptools version and silently stop resolving after a toolchain bump. Prefer a symlink or copy
inside the package directory. Candidate for a pre-commit grep.

## build.feature-gate-version-threshold (important, new/edited `_CCCL_HAS_*`/version-threshold feature-availability macros)

<!-- provenance:
  #3114→#3136 _CCCL_HAS_PDL defined as CTK>=11.8 with no citation, actual minimum is CTK 12.0;
  #3623→#3644,#3654 untested "nvrtc on windows" fix guessed an MSVC-frontend proxy condition instead of the real NVRTC-version requirement for __builtin_isfinite;
  reverted wholesale, fixed with a version check;
  #7339→#7838,#7856 cudaDevAttrHostNumaMemoryPoolsSupported gated at CTK>=12.6 though the enum exists only from 12.9;
  #8283→#8357 uncited _CCCL_STD_VER>=2023 gate stacked on _CCCL_HAS_CPP_ATTRIBUTE(assume), needlessly disabling [[assume]] pre-C++23;
  #10997→#11031,#11033 __remove_cv builtin broadened to NVCC/GCC with only a libstdc++-version gate and no NVCC-version floor, miscompiled on NVCC 12.9/13.0
-->

<!-- note:
    I don't think it's feasible to add such a rule.
    Or can we ask the agent to entertain a web search and study the corresponding manuals?
-->

When a diff adds or edits a capability macro gating a CUDA-toolkit- or compiler-version-dependent
feature (programmatic dependent launch, a `__builtin_*`, cluster launch, …) behind a numeric threshold
(`_CCCL_CUDACC_AT_LEAST(11, 8)`, `_CCCL_COMPILER(NVRTC, >, 12, 2)`), require the diff to carry a
citation (CUDA Programming Guide section, release notes, linked issue) proving that number is the
feature's actual minimum. CI only exercises the toolkit versions in the matrix, almost never the exact
boundary, so an off-by-one-release threshold ships silently. Flag proxy conditions the same way (e.g.
"is this an MSVC-flavored frontend" as a stand-in for "is the version new enough") — treat an uncited
threshold as a required-changes item. The same applies to a C++ dialect threshold
(`_CCCL_STD_VER >= 20XX`) stacked ON TOP OF an existing feature-test macro
(`_CCCL_HAS_CPP_ATTRIBUTE(...)`, `__has_builtin`): the feature-test macro is usually already the
complete per-compiler answer, so an uncited dialect guard on top silently disables the feature for
every dialect and compiler that genuinely supports it. Likewise when a diff BROADENS a
`_CCCL_BUILTIN_*`/`_CCCL_HAS_BUILTIN` availability condition to a new vendor family (e.g. enabling a
type-trait builtin for NVCC/GCC that was clang-only): require an explicit per-compiler version floor
(`_CCCL_CUDA_COMPILER(NVCC, >=, X, Y)`) and a statement of which versions were tested — NVCC has
shipped release-specific miscompiles of individual builtins, and PR CI typically builds only the
newest toolchain.

## build.mixed-cuda-runtime-linkage (important, CMake changes linking CUDA runtime libraries)

<!-- provenance:
  #6122→#7221 cudax catch2 tests linked shared CUDA::cudart while c2h and CMake's default CUDA_RUNTIME_LIBRARY pull in cudart_static, breaking the MSVC link;
  MSVC cudax CI was disabled meanwhile (#7139) (pair auto-inferred as #7139→#7221)
-->

<!-- note: again looks like a too specific failure, which went undetected at the PR because the CI was disabled for reasons. It was caught by the CI later -->

When a diff adds or edits `target_link_libraries` with `CUDA::cudart` (shared) or a bare `cudart`,
check which runtime flavor everything else in the final link closure uses: CMake's default
`CUDA_RUNTIME_LIBRARY` is Static, so any target compiling CUDA sources (including helper libraries
like test harnesses) already pulls in `cudart_static`. Mixing shared and static CUDA runtimes in one
binary links silently on Linux but fails with duplicate-symbol errors under MSVC — which PR CI may not
cover for the touched project. Require one consistent flavor per binary (in CCCL:
`CUDA::cudart_static`). Candidate for a pre-commit grep.

## build.host-feature-unavailable-in-device (important, code compiled for both host and device)

<!-- provenance:
  #9777→#10508 single-pass host+device fpemu test instantiated fpemu<_Float64> guarded only by __STDCPP_FLOAT64_T__, breaking nvcc device compilation under C++23
-->
<!-- note:
  CI could only catch this with a C++23 job (nvcc -std=c++23 + libstdc++ 13+); the matrix builds
  17/20 only. Even then, each new dialect/stdlib pairing reopens the gap, so the rule stays relevant.
-->

When a diff adds a code path selected purely by a *language* feature-test macro (`__cpp_*`,
`__STDCPP_*_T__`, C++23 `<stdfloat>` types) inside a function compiled for BOTH host and device (a
`_CCCL_HOST_DEVICE` function reachable from a `__global__` kernel, a single-pass host+device test),
check whether nvcc's *device* front end actually supports the construct. The feature-test macro
reflects the host compiler; nvcc's device compilation can lag and reject the construct in `__device__`
context (e.g. nvcc does not support C++23 extended-precision scalar types in device code even when the
host defines `__STDCPP_*_T__`). Require an additional compiler check
(`&& !_CCCL_CUDA_COMPILER(NVCC)`) or a host-only path guarded by `!_CCCL_CUDA_COMPILATION()`.

## build.tile-incompatible-api-macro (important, CCCL headers using `_CCCL_API`/`_CCCL_TRIVIAL_API`/`_CCCL_NODEBUG_API`)

<!-- provenance:
  #9777→#10538 fpemu asm-heavy arithmetic helpers marked _CCCL_API instead of _CCCL_HOST_DEVICE_API (pair auto-inferred as #10533→#10538;
  #10533 was a closed draft, real intro via git blame);
  #9752→#10539 CUB tuning to_string() debug helpers marked _CCCL_API (pair auto-inferred as #10533→#10539);
  #8475→#9191 cuda::std::simd complex support (inline asm, manual unrolling) marked _CCCL_API, breaking tile-mode compilation
-->

<!-- note:
  A tile-mode build job would catch this class mechanically (hard compile errors, e.g. "asm statement
  is unsupported in tile code"); lit has an enable-tile feature but it is off by default and not in the
  PR matrix. Coverage is limited by lazy instantiation of templates. Retire this rule once such a job
  exists.
-->

When a diff marks a function with the generic `_CCCL_API` (or `_CCCL_TRIVIAL_API`/`_CCCL_NODEBUG_API`)
macro, check whether the body relies on a construct tile-mode compilation cannot support — inline
`asm`, or a pure host-side debug/diagnostic helper (`to_string`, a formatter). `_CCCL_API` silently
opts the function into `__tile__` compilation; `_CCCL_HOST_DEVICE_API` is host+device but excluded from
tile mode. Both compile identically outside a tile-mode CI job, which most PR CI does not run. Flag any
newly `_CCCL_API`-marked function containing inline `asm` or whose only purpose is host-side
stringification; require `_CCCL_HOST_DEVICE_API` (or narrower).

## build.transitive-include-removal (important, header/include refactors spanning many files)

<!-- related: correctness.stale-refs-after-rename covers the rename-flavored variant of out-of-diff breakage -->
<!-- provenance:
  #8216→#10813 centralizing STL stream includes removed a transitive <iostream> that the README.md quick-start example (never touched by the diff) relied on
-->

<!-- note:
  This seems like we should just have more documentation examples unit tested.
-->

When a diff centralizes or minimizes `#include`s across many library headers, check whether it removes
a standard-library include (`<iostream>`, `<sstream>`) that was previously available *transitively*
through public headers. Search README/getting-started snippets, `examples/`, and documentation code
blocks for use of the now-unavailable facility (`std::cout`, …) — including files the diff never
touches — against the diff's base revision, not a possibly already-patched checkout. The cleanup is
usually right for the library's own tests (which the diff fixes up), but prose-embedded code samples
rely on the same transitive includes and are not compiled by CI.

## correctness.partial-optimizer-workaround (important, compiler/optimizer-bug workarounds in CUDA kernels)

<!-- provenance:
  #8656→#8839 asm-volatile barrier for an NVCC codegen bug applied to one masked-copy pointer group in squadStoreBulkSync but not the structurally identical doStartCopy block below, causing an illegal memory access on sm_121 (intro corrected from issue #8838)
-->

<!-- I don't think this can be generalized. Compiler workarounds have to be re-evaluated at every site they are used -->

When a diff adds a compiler/optimizer-bug workaround (an `asm volatile("" : "+r"(x))` barrier,
`#pragma`, `volatile`) to guard specific values against a cited codegen bug, search the surrounding
function and structurally similar siblings for OTHER occurrences deriving/consuming a value the same
way (a masked store from the same shifted-pointer arithmetic, a similar loop-carried value) and check
whether they need the identical barrier. A workaround applied to one of several structurally-identical
occurrences fixes the reported symptom but leaves the siblings silently exposed to the same miscompile.

## correctness.hidden-friend-broad-constraint (important, hidden-friend operator/function templates defined inline in a class template)

<!-- provenance:
  #10687→#10723 constant_wrapper's hidden-friend operators, constrained only by a broad duck-typed trait after a narrower documented-workaround constraint was removed, were picked up by nvcc on Windows for unrelated classes via ADL, breaking main
-->

<!-- note: this seems niche and I don't have a good idea how to generalize it -->

When a diff removes or weakens the SFINAE/`requires` constraint on a hidden-friend (ADL-only) operator
template defined inline inside a class template — replacing a check that the parameter is the enclosing
template with only a broader duck-typed trait (e.g. "has a static constexpr `value` member") — flag it.
Hidden friends are found via ADL for every type satisfying the broader constraint; on some compilers
(notably nvcc on Windows during host-stub generation) this picks up unrelated classes, producing
invalid overload sets that break files the diff never touched. A removed comment documenting why the
narrower constraint existed is a strong signal to re-verify against the full compiler matrix. Prefer
keeping an explicit identity check alongside any broader constraint.

## correctness.sfinae-hides-typo (important, new type-trait specializations or overload constraints using SFINAE/`enable_if`/`requires`)

<!-- provenance:
  #2756→#9020 (pair auto-inferred from issue #1877) is_fixed_size_random_access_range's mdspan specialization wrote E::rank == 1 instead of E::rank() == 1, silently dropping the specialization from SFINAE matching;
  hidden because the mdspan tests were gated behind a stricter dialect check than the implementation
-->

<!-- I don't see how we can automatically detect spelling errors in member detection idoms in general -->

When a diff adds a type trait or overload constrained via SFINAE (`enable_if_t<…>`, a partial
specialization's template-argument expression, a `requires` clause) referencing a dependent type's
static member, verify the member is used the way that type's API defines it — a function (`E::rank()`)
vs. a value. A missing `()` raises no compile error inside a SFINAE context: the substitution becomes
ill-formed and the specialization silently drops out of matching instead of erroring. Mixing
`E::rank == 1` with `E::rank_dynamic() == 0` in one expression is a red flag. Also check the new path
is exercised by tests gated at the same (or looser) feature threshold as the implementation — a test
behind `#if _CCCL_STD_VER >= 2023` for code gated at `>= 2017` leaves the range in between untested.

## api.iterator-tag-propagation (critical, replacements of iterator types used in Thrust call chains)

<!-- provenance:
  #5839→#5925 swapping thrust::counting_iterator for cuda::counting_iterator lost any_system_tag under transform_iterator wrapping, selecting the wrong backend (issue #5860); reverted
-->

<!-- I think this rule is too specific to the thrust iterator migration. once completed, we don't need the rule -->

When a diff replaces an iterator type across call sites (e.g. a `thrust::` fancy iterator for a
`cuda::` counterpart), verify its iterator category and system tag still propagate correctly when the
iterator is wrapped by ANOTHER fancy iterator (`thrust::transform_iterator`, `zip_iterator`, …): a
category passed through unchanged instead of recomputed can silently downgrade `any_system_tag` to a
concrete system, selecting the wrong backend layers away from the swap. Require a test exercising the
new iterator through at least one other fancy-iterator wrapper, not only standalone.

## api.namespace-shadowing (important, public headers introducing new types/aliases)

<!-- provenance:
  #2343→#2372 cuda::experimental::stream_ref shadowed ::cuda::stream_ref, breaking unqualified uses in cudax/samples/vector_add (a file not touched by #2343)
-->

<!-- note:
  I don't think we can add this rule, since it is a legitimate situation. For example, we may add an
  entity as cuda::experimental::XXX, and later expose it as cuda::XXX. But to not break users of the
  experimental facility immediately, we may decide to keep it (and for example deprecate it).
-->

Flag a new type or alias introduced into a nested/inner namespace whose unqualified name collides with
an existing type from an enclosing namespace (e.g. `namespace cuda::experimental { struct stream_ref :
::cuda::stream_ref {…}; }` shadowing `::cuda::stream_ref`). Code that used the unqualified name from
within the inner namespace — including samples, tests, and downstream code — silently rebinds to the
new type. Grep the repo (including examples/samples) for unqualified uses of the shadowed name in
scopes affected by the new declaration; require qualification or confirm no ambiguity results.

## api.cross-project-name-collision (important, new or renamed public names in cuda/cub/thrust namespaces)

<!-- provenance:
  #5153→#6927 (backport #6937) cudax launch-argument utility renamed __launch_transform→device_transform, colliding with the cub::DeviceTransform algorithm family;
  renamed back after release branching (pair auto-inferred as #6927→#6937)
-->

<!-- note: not choosing a name that collides with a different API, but compiles fine, is not
  something we can automatically detect. This is about API design, for which I think I don't want
  to have AI input -->

When a diff introduces or renames a public (non-`__`-prefixed) symbol — especially when promoting an
internal `__detail` utility to a public name — grep every CCCL subproject (cub, thrust, libcudacxx,
c/parallel, python/cuda_cccl) for the same name or its CamelCase/snake_case variants. Flag a new name
colliding with an established algorithm family elsewhere in CCCL (e.g. a `cuda::device_xxx` utility
whose name matches `cub::DeviceXxx` performing a *different* operation): the collision confuses users
and blocks the name from later being used for the consistent counterpart API, forcing a breaking
rename after release. A PR comment like "I'm not attached to the name" is a signal to run this check.

## api.cross-subproject-signature-change (important, public entry points in cub/thrust/libcudacxx consumed by other CCCL subprojects)

<!-- provenance:
  #4451→#4498 thrust::cuda::par.on(...) narrowed from cuda::stream_ref back to cudaStream_t, breaking cudax call sites relying on the conversion chain;
  PR CI missed it because ci/inspect_changes didn't list cudax as depending on thrust/cub
-->

When a diff changes the parameter or return type of a widely-used public entry point — especially
swapping a vocabulary/wrapper type (`cuda::stream_ref`) for a narrower concrete type (`cudaStream_t`)
or vice versa — grep the ENTIRE repository for call sites in OTHER CCCL subprojects (cudax, c/parallel,
examples, benchmarks), against the diff's BASE revision (the checkout may already contain a later
downstream fix that makes the grep come back clean). C++ applies only one user-defined conversion, so a
wrapper type two conversions away (wrapper → OldType → NewType) stops compiling at the call site, not
at the declaration under review. Green PR CI is not sufficient: `ci/inspect_changes` decides which
subprojects get rebuilt, and a missing dependency edge means a real downstream break compiles clean.

## api.trait-specialization-member-shape (important, new specializations of standard-library customization-point traits)

<!-- provenance:
  #7439→#8486,#8488 thrust device_ptr/pointer/normal_iterator pointer_traits specializations defined rebind as a nested struct instead of an alias template;
  allocator_traits::rebind<U> silently named the struct, breaking RAPIDS via rmm's thrust_allocator
-->

When a diff adds a specialization of a standard-library trait that generic code consumes structurally
(`pointer_traits`, `allocator_traits`, `iterator_traits`), verify the member has the exact same SHAPE
as the primary template's: an alias template stays an alias template, not a nested struct wrapping
`using other = …`. A nested `struct rebind { using other = …; };` compiles fine and even compiles where
used (`rebind<U>` just names the struct), so the mismatch is invisible in the introducing PR's own
tests and only manifests layers away in a generic consumer, often in third-party code. Diff any new
trait specialization against the primary template's declaration to confirm identical member shape, and
add a `static_assert` exercising the trait exactly as its documented consumer does.

## api.duplicated-derived-type-not-updated (important, CUB/Thrust algorithms with multiple public overload families layered over one dispatch)

<!-- provenance:
  #9289→#9676 DeviceReduce's env-based overloads in device_reduce.cuh independently recomputed accum_t via __accumulator_t instead of delegating to detail::reduce::select_accum_t, so the new no_init_t sentinel wasn't supported by the env overload family
-->

When a diff adds a new special-case/sentinel argument (a `no_init_t`-style marker) supported via a
shared low-level type-computation helper (`select_*_t`), grep every other public overload family of
the same algorithm for a second, independently-written computation of the same derived type (an inline
`using accum_t = __accumulator_t<…>` instead of the shared helper). A sibling overload family that
recomputes the type inline silently fails to support the new case — typically a hard compile error the
moment a caller passes the sentinel there. Check EVERY public entry point, especially the
`device_*.cuh` facade header layered over the `dispatch_*.cuh` the diff touches — it is frequently not
in the diff at all; open it anyway, against the diff's base revision (the checkout may already contain
a later fix). Prefer replacing duplicated formulas with the shared helper.

## perf.tuning-domain-vs-benchmark-coverage (important, CUB/Thrust tuning-policy tables and arch/type/op-gated perf constants)

<!-- provenance:
  #8611→#8689 I8 lookahead-scan tuning chosen from aggregate search scores regressed small sizes 20%, reverted;
  #8352→#9813 (backport #9848;
  pair auto-inferred as #9813→#9848) tuned entries gated op==plus left custom-op scans on slow fallback (issue #9811);
  #10923→#10943 unbounded cc>={10,7} Rubin bytes-in-flight captured untested sm_120+
-->

When a diff adds or changes tuning values or the conditions selecting them (`tuning_*.cuh` policy
selectors, per-arch perf constants), compare the gate's domain against what was actually benchmarked,
in both directions:
- Too wide: an arch gate with only a lower bound (`cc >= {X,Y}`) atop an if-chain also captures every
  future architecture; bound it to the tested family or justify why it generalizes.
- Too narrow: tuned entries gated on an operation kind or type category (e.g. only `op == plus`) send
  every other case to a fallback policy; state what those cases get and how it performs vs. before.
- A tuning value justified only by an aggregate tuning-search score hides per-problem-size
  regressions; require a per-size benchmark table from small (~2^16) through large (≥2^28) inputs,
  especially when replacing a previously shipped tuning.

## perf.jit-cache-key-vs-codegen-inputs (important, cuda.compute / c.parallel build-result caching)

<!-- provenance:
  #7657→#9596 (pair auto-inferred as #9475→#9596) histogram build cache keyed on runtime lower/upper level values and exact num_samples, forcing a recompile per distinct bounds (issue #9594)
-->

When a diff constructs or changes the cache key of a memoized JIT/build result (`@lru_cache`-style
`_make_*_impl` builders, `cache_build_results` keys), require the key to consist of exactly the inputs
that affect the generated code: dtypes, iterator kinds, operator identity, regime flags. Flag
runtime-only kernel arguments in the key — scalar bound/init/seed values, exact element counts — since
every distinct runtime value then triggers a full recompile, silently destroying cache hit rates.
Conversely, flag a key that omits a compile-affecting input, which causes wrong-kernel reuse.

## perf.intrinsic-wrapper-codegen-parity (important, new generic wrappers over device intrinsics in libcudacxx/cub)

<!-- provenance:
  #3907→#10035 (pair auto-inferred as #8391→#10035) cuda::device::warp_shuffle memcpy-punned values through uninitialized locals and recomputed the predicate, inflating register pressure vs raw __shfl intrinsics
-->

When a diff adds a generic (any-type) wrapper over a hardware intrinsic (warp shuffle/vote/match,
atomics) that round-trips values through `memcpy` into local arrays or structs, flag it unless the PR
ships a codegen comparison (SASS/PTX test) showing parity with the raw intrinsic for common types.
`memcpy` into uninitialized temporaries and per-32-bit-word loops inflate register pressure with no
functional symptom, making every consumer slower than hand-written intrinsics. Initialize the
destination before `memcpy`, prefer `bit_cast`-style punning, and do not recompute outputs the
instruction already produces (e.g. the shuffle predicate).

## test.hidden-consumer-of-internal-api (important, python bindings/internal refactors)

<!-- provenance:
  #4353→#4508 new radix_sort Cython binding declared its build-result struct with no fields and no _get_cubin(), breaking the SASS diff test harness;
  #6938→#6999 operator-handling/caching refactor broke benchmark scripts calling algorithm.cache_clear() directly
-->

Internal surfaces (Cython/C build-result classes, caching decorators, op-handling wrappers) are relied
on by tooling outside the unit-test tree — SASS/PTX diff harnesses, benchmark scripts, examples — via
specific attributes, not the public algorithm API. Flag: (1) a new binding class that mirrors existing
siblings but omits a boilerplate member the siblings have (declaring the underlying `extern` struct
with `pass`/no fields is a tell); (2) a refactor of a shared caching/build-result mechanism that
changes how external code accesses it. Grep `benchmarks/`, SASS-test, and example directories for
consumers of the old interface; a PR checklist admitting "no new tests" for such a change is a signal
to check harder, not to skip.

## test.python-array-init-construction (important, python tests)

<!-- provenance:
  #3216→#4251 np.zeros([0], dtype=dt) used as a scalar init value across ~20 parametrized cache-identity tests, actually producing a zero-length array
-->

When python test code constructs a host-side "init value" or single-element array (commonly passed as
a scalar seed/identity argument to a reduce-like API), check the construction actually produces the
intended contents. `numpy.zeros(shape, …)` takes a *shape*, not a *value*: `np.zeros([0], dtype=dt)`
creates a length-zero array, not a one-element array containing 0 — a plausible typo for
`np.zeros(1, dtype=dt)` or `np.array([0], dtype=dt)`. Easy to miss when copy-pasted across many
parametrized cases that only compare object identity. Candidate for a pre-commit grep.

## test.sibling-config-drift (important, CMake test configuration)

<!-- provenance:
  #4802→#5242 _CCCL_HEADER_TEST added to libcudacxx/test/internal_headers/CMakeLists.txt only, silently skipping the new prologue/epilogue check for public_headers and public_headers_host_only
-->

CCCL often configures near-identical test targets from multiple near-duplicate CMakeLists.txt files
(internal-header, public-header, host-only public-header tests). When a diff adds a new
`target_compile_definitions`/compile flag/macro to one such test-configuration file, check every
sibling CMakeLists.txt defining a structurally similar target for the same addition. Adding a gating
macro to only one of several near-duplicate test targets silently skips the check for the others.

## test.new-arch-coverage-gap (important, CUB/Thrust architecture-conditional dispatch/tuning code)

<!-- provenance:
  #8922→#8972 SM120-conditional codegen workaround added to CUB scan tuning with no CI job change;
  the C Parallel Library's light PR CI only builds SM75, so the SM120 path went unverified;
  #10008→#10047 sm_107 added to nv/target without a matching GEN_POLICY entry in CUB's exhaustive ChainedPolicy self-test (NVBug 6441160)
-->

When a diff adds architecture-conditional branching keyed on a specific SM/compute-capability literal
(a new `nv::target::sm_XXX` selector, an arch bucket in tuning/dispatch code) or a brand-new
architecture macro, check whether the SAME diff includes a test-side or CI-config change that
exercises that architecture: a test restricted to that SM, a `ci/matrix.yaml` row/GPU covering it, or
an entry in an exhaustive self-test table enumerating every known architecture (e.g. CUB's
`policy_hub_all` in `catch2_test_util_device.cu`). Do not assume existing CI covers it: dependent
projects' "light" PR jobs are pinned to a single default SM. Flag the omission; ask the author to
point at the specific job/test covering the new architecture, or add one.

## infra.ci-flag-removal (important, `ci/*.sh`, `ci/matrix.yaml`, and other shared automation/config)

<!-- provenance:
  #493→#1458 removed -disable-benchmarks / ENABLE_CUB_BENCHMARKS env-var override when refactoring ci/build_cub.sh;
  #7919→#10057 (via prerequisite #8160) a PR titled "Remove CuPy upper bound" also silently dropped 12.0 from ctk: lists in ci/matrix.yaml, cutting CTK 12.0 python CI coverage unnoticed for months (issue #8156);
  #4924→#5543 release-wheels.yml simplification dropped the -p "*${comp}*" filter from gh run download, breaking wheel releases (pair auto-inferred as #5541→#5543)
-->

When a diff touches `ci/*.sh` or `ci/matrix.yaml`, check whether it silently removes an existing
override mechanism or coverage value that other automation may depend on — a CLI flag or
`${VAR:=default}` escape hatch in a script, or a specific version value (`ctk`, `cxx`, `py_version`,
`sm`) dropped from a job row's value list in the matrix. Cross-check the removal against the diff's
stated purpose: if the PR title/description is about something unrelated and a matrix row's value list
shrank as a side effect, treat it as likely-unintentional — a silent coverage drop produces no CI
failure and can go undetected for a long time. Preserve the override, or call out and justify the
removal in the same change.

## infra.wire-new-routing-value (important, CI workflow/matrix/config files under `.github/` and `ci/`, pre-commit config)

<!-- provenance:
  #1206→#1324 windows matrix split left WINDOWS_* job outputs unconsumed, silently dropping MSVC PR coverage;
  #2007→#2009 workflow_dispatch trigger added but build-rapids.yml's event_name gate wasn't updated;
  #3168→#3182 [tool.codespell] added to pyproject.toml but the pre-commit hook never passed --toml to read it;
  #3572→#3580 testing: runner flags generated a label with no backing pool, stalling PRs;
  #3739→#4792 DispatchScan template-parameter reorder left the exclusive-scan benchmark instantiating positionally, binding policy_t to the wrong slot;
  #4476→#4478 Slack channel rerouted in the wrong workflow file;
  #5898→#5967 devcontainer dir renamed cuda12.9→13.0 but the job matrix cuda: value in the same file kept 12.9;
  #6164→#6169 new cmake_options: matrix key had no -cmake-options parameter in the Windows scripts
-->

<!-- my understanding of these kinds of bugs and whether we can enforce a check here is too limited -->

When a diff adds, renames, or splits a config value that other files must recognize, verify every
consumer was updated in the same diff — "I added the setting" does not mean "the tool now uses it":
- a new/renamed GitHub Actions job `output` must be read via `needs.<job>.outputs.<name>` somewhere,
  and a rename must update *every* reader (Linux updated but Windows left on the old name is a red flag);
- a new trigger in a workflow's `on:` block must be recognized by every `github.event_name` conditional
  downstream (including composite actions), or the new trigger silently runs a crippled job subset;
- a new `[tool.X]` section in `pyproject.toml` meant for a pre-commit hook is inert unless the hook
  entry passes it (e.g. `args: ["--toml", "pyproject.toml"]` + parser dependency);
- a new/changed runner label or pool selector must map to a pool that exists — a job queuing forever
  for a nonexistent runner does not show as a CI failure, so green CI is not sufficient;
- when a version embedded in a directory/image/config name is bumped (e.g. `cuda12.9-conda/` →
  `cuda13.0-conda/`), grep for every other place that version string is duplicated as a matrix value or
  interpolated path — including within the same file the diff already touches in an unrelated hunk; a
  partial bump leaves some jobs silently selecting the old, no-longer-existing directory or image;
- when a diff adds a job-matrix field or CLI option that per-OS runner scripts must translate (e.g. a
  new `cmake_options:` matrix key), confirm *every* platform's scripts declare and forward the new
  parameter, not just the platform the author tested locally.

## infra.mechanical-diff-outliers (important, bulk regenerated/scripted diffs: devcontainers, CI matrices, sweeps)

<!-- provenance:
  #1935→#1955 CUDA 12.5 devcontainer bump deleted the hand-maintained RAPIDS devcontainer symlink with no replacement, breaking RAPIDS CI
-->

<!-- I can imagine this scenario, but a review agent may be to noisy about. I am not sure whether we should add it. We can also add and remove if it gets too annoying. -->

A bulk mechanical diff (regenerated file set, scripted sweep) shows one dominant, uniform
transformation; review the outliers, not the bulk: deletions with no regenerated counterpart, files
matching the pattern that were NOT touched, or hunks transformed differently from the rest. For each
outlier, two checks decide: does anything still reference the deleted/diverging name (grep
`.github/workflows/`, `ci/`, `.devcontainer/`) — surviving references mean breakage; and does the
diff's stated purpose account for it (e.g. "drop gcc-9" explains vanished gcc-9 entries) — an
unexplained outlier is usually a hand-maintained special case the generating tool doesn't know about.

## infra.version-bump-behind-ci-skip (important, pre-commit hook / pinned CI-tool version bumps)

<!-- provenance:
  #10729→#10876 mirrors-mypy bumped v1.16.1→v2.1.0 while mypy sat on the pre-commit.ci skip: list, so green CI never exercised the new version;
  it then crashed locally on a config older mypy tolerated
-->

<!-- not sure this rule is worth the effort -->

When a diff bumps the pinned version of a pre-commit hook (a `rev:` change in
`.pre-commit-config.yaml`), do not stop at the hunk: read the top-level `ci: skip: [...]` list as it
existed at the diff's base commit (the skip list is usually untouched by the bump, so it never appears
in the hunk — and the current checkout may already contain a later cleanup). If the bumped hook id is
on that list, green pre-commit.ci output is not evidence the new version works — the hook never ran.
Require either lifting the skip (even temporarily) to prove the bumped tool passes, or explicit
confirmation it was run locally against every config section it reads.

## docs.migration-guide-deprecation (important, public macros/functions/classes in cub/thrust/libcudacxx and `docs/cccl/*_migration_guide.rst`)

<!-- provenance:
  #4165→#4237 CUB macro deprecations missing from migration guide;
  #4165→#4242 same fix backported to branch/3.0.x
-->


<!-- that's not a bad idea, but we can also easily collect the list of deprecated symbols from grepping through the source code -->

When a diff removes, renames, or deprecates (e.g. `_CCCL_DEPRECATED`) a public macro, function, class,
or type, verify that `docs/cccl/*_migration_guide.rst` gains a matching entry describing the removal
and its replacement (or "No replacement"). Applies equally to mainline PRs and backport PRs. Flag a
diff that deletes/deprecates public symbols across `.cuh`/`.h` files without any corresponding
migration-guide update.

## Retired

<!-- provenance: #651, #1257 -->

- `if constexpr` fallback instantiation firing `static_assert`s pre-C++17: obsolete since
  C++17 is the minimum dialect.
