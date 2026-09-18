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

## build.cuda-mode-detection (critical, all C++ headers)

<!-- provenance:
  #1199→#1224 nvc++ -stdpar fully broken
-->

Flag raw `__CUDACC__`, `__NVCOMPILER`, `__CUDA_ARCH__`, or other vendor macros outside the
CCCL config headers; use `_CCCL_CUDA_COMPILATION()`, `_CCCL_COMPILER(...)`,
`_CCCL_DEVICE_COMPILATION()`. NVC++ `-stdpar` enables CUDA without `__CUDACC__`.
Candidate for a pre-commit grep.

## build.deprecation-default-mismatch (important, CMake/config changes)

<!-- provenance:
  #2166→#2217 added C++14 deprecation warning without bumping CUB/Thrust's own default CMake dialect off C++14
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

Flag packaging metadata (`license_files=`, `data_files=`, `package_data=`, `[tool.setuptools]` file
lists, `MANIFEST.in` includes) referencing a path outside the package's own directory via `../`.
Packaging tools only reliably include files inside the package/sdist root; an escaping path can work
on one setuptools version and silently stop resolving after a toolchain bump. Prefer a symlink or copy
inside the package directory. Candidate for a pre-commit grep.

## build.stale-compiler-macro (important, all C++ preprocessor guards)

<!-- provenance:
  #2518→#2904 for_each_in_extent added defined(_CCCL_COMPILER_GCC) && _CCCL_GCC_VERSION < 100000, a pre-migration macro pairing that no longer exists, silently flipping the guard permanently true;
  #2749→#2989 same stale _CCCL_GCC_VERSION pattern disabled _CCCL_BUILTIN_LAUNDER on all GCC versions
-->

When a diff adds a NEW compiler-identity or compiler-version preprocessor guard, flag any spelling
that does not use the current unified helper (`_CCCL_COMPILER(<VENDOR>, <op>, <ver>)`,
`_CCCL_CUDACC_AT_LEAST(...)`) — in particular a bare `defined(_CCCL_COMPILER_<VENDOR>)` combined with
a separate `_CCCL_<VENDOR>_VERSION <op> <NNNNNN>` macro. That older two-macro convention has been
migrated away from; if the version macro no longer exists, it silently evaluates to `0` inside `#if`,
flipping an intended "only on old GCC" guard into "always true" on every compiler, which can leak
GNU-only extensions into MSVC/NVHPC/Clang builds. Confirm the referenced macro is still defined at the
diff's base revision. Candidate for a pre-commit grep.

## build.feature-gate-version-threshold (important, new/edited `_CCCL_HAS_*`/version-threshold feature-availability macros)

<!-- provenance:
  #3114→#3136 _CCCL_HAS_PDL defined as CTK>=11.8 with no citation, actual minimum is CTK 12.0;
  #3623→#3644,#3654 untested "nvrtc on windows" fix guessed an MSVC-frontend proxy condition instead of the real NVRTC-version requirement for __builtin_isfinite;
  reverted wholesale, fixed with a version check;
  #7339→#7838,#7856 cudaDevAttrHostNumaMemoryPoolsSupported gated at CTK>=12.6 though the enum exists only from 12.9;
  #8283→#8357 uncited _CCCL_STD_VER>=2023 gate stacked on _CCCL_HAS_CPP_ATTRIBUTE(assume), needlessly disabling [[assume]] pre-C++23;
  #10997→#11031,#11033 __remove_cv builtin broadened to NVCC/GCC with only a libstdc++-version gate and no NVCC-version floor, miscompiled on NVCC 12.9/13.0
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

## build.narrow-multiply-then-widen (important, C++/CUDA code computing a size/count/capacity/offset, including test files)

<!-- provenance:
  #7705→#9736 fixed_capacity_map tests computed capacity via static_cast<size_t>(num_keys * 2), multiplying in int before widening, breaking a clang-tidy CI check (pair auto-inferred as #9719→#9736)
-->

When a diff computes a size, count, capacity, or offset by multiplying two narrower-typed operands and
only widens the RESULT afterward — `static_cast<SizeT>(a * b)` — flag it: the multiplication executes
in the narrower type and can silently overflow before the cast runs. Require widening an operand
first: `SizeT{a} * b`. Check every occurrence in the diff, including test/benchmark files sizing a
buffer from a small `int` loop-count variable — these are not exempt merely because the test's inputs
happen to be small. Candidate for a pre-commit grep.

## build.long-not-64bit-on-windows (important, C++/CUDA code aliasing or computing with `long`)

<!-- provenance:
  #6068→#6081 c/parallel three_way_partition used `using OffsetT = long;
  `, whose choose_signed_offset static_assert fails under MSVC (LLP64: long is 32-bit)
-->

Flag `long` used as a stand-in for a 64-bit or pointer-width integer type (`using X = long;`, casts to
`long`, `long` chosen for an offset/size/count spanning the address space) unless guarded for MSVC.
Windows targets (MSVC, clang-cl) use the LLP64 data model where `long` is 32 bits, unlike LP64
Linux/macOS. Code assuming `long == int64_t` builds and passes on Linux CI but silently truncates or
fails a `static_assert` on Windows. Prefer `ptrdiff_t`, `int64_t`, or `long long`. Candidate for a
pre-commit grep.

## build.mixed-cuda-runtime-linkage (important, CMake changes linking CUDA runtime libraries)

<!-- provenance:
  #6122→#7221 cudax catch2 tests linked shared CUDA::cudart while c2h and CMake's default CUDA_RUNTIME_LIBRARY pull in cudart_static, breaking the MSVC link;
  MSVC cudax CI was disabled meanwhile (#7139) (pair auto-inferred as #7139→#7221)
-->

When a diff adds or edits `target_link_libraries` with `CUDA::cudart` (shared) or a bare `cudart`,
check which runtime flavor everything else in the final link closure uses: CMake's default
`CUDA_RUNTIME_LIBRARY` is Static, so any target compiling CUDA sources (including helper libraries
like test harnesses) already pulls in `cudart_static`. Mixing shared and static CUDA runtimes in one
binary links silently on Linux but fails with duplicate-symbol errors under MSVC — which PR CI may not
cover for the touched project. Require one consistent flavor per binary (in CCCL:
`CUDA::cudart_static`). Candidate for a pre-commit grep.

## build.windows-min-max-macro (important, C++ code calling `.max()`/`.min()` or naming a new member/trait `max`/`min`)

<!-- provenance:
  #8875→#9246 argument-annotation trait member named max, computed via unparenthesized numeric_limits<T>::max(), a preprocessor argument-count error under <windows.h>'s max/min macros;
  renamed to highest/lowest and parenthesized
-->

Flag an unparenthesized call to a function literally named `max` or `min` (most commonly
`std::numeric_limits<T>::max()`), and a newly introduced member or trait literally named `max`/`min`.
On Windows, `<windows.h>` (pulled in transitively unless `NOMINMAX` is defined first) `#define`s
`max`/`min` as 2-argument function-like macros, so a zero-argument `::max()` is a hard preprocessor
error — only on MSVC/Windows configurations, often missing from local builds. Require the macro-safe
spelling `(std::numeric_limits<T>::max)()`, and prefer naming new members something other than bare
`max`/`min` (e.g. `highest`/`lowest`). Candidate for a pre-commit grep.

## build.host-feature-unavailable-in-device (important, code compiled for both host and device)

<!-- provenance:
  #9777→#10508 single-pass host+device fpemu test instantiated fpemu<_Float64> guarded only by __STDCPP_FLOAT64_T__, breaking nvcc device compilation under C++23
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

When a diff centralizes or minimizes `#include`s across many library headers, check whether it removes
a standard-library include (`<iostream>`, `<sstream>`) that was previously available *transitively*
through public headers. Search README/getting-started snippets, `examples/`, and documentation code
blocks for use of the now-unavailable facility (`std::cout`, …) — including files the diff never
touches — against the diff's base revision, not a possibly already-patched checkout. The cleanup is
usually right for the library's own tests (which the diff fixes up), but prose-embedded code samples
rely on the same transitive includes and are not compiled by CI.

## correctness.header-kernel-odr-template-hack (critical, `__global__` kernels and free functions defined in headers)

<!-- provenance:
  #2641→#2656 templatized CUDASTF's callback_completion_kernel to dodge a multiple-definition linker error, risking runtime launch errors
-->

Flag a `__global__` function (or any function) defined in a header that is turned into a template
(especially with an unused/default-only parameter like `template <int = 0>`) purely to work around a
"multiple definition" / ODR linker error. This is not equivalent to `inline`: template kernels
instantiated identically in multiple TUs can still misbehave at kernel-launch time. The correct fix is
`inline` for ordinary functions and `static` or an unnamed namespace for `__global__` kernels. A
comment like "template to make this inline without using inline" is a strong signal.

## correctness.remove-equivalence-sanity-check (critical, any diff)

<!-- provenance:
  #3743→#3866 dropping deprecated cub::Traits CATEGORY usage also deleted the static_assert cross-checks (old_IS_SMALL_UNSIGNED, "sanity check, remove eventually") comparing new type classification to the old one, breaking dispatch for library-extended types;
  NVBug 5121653
-->

If a diff deletes a `static_assert` (or other check) that cross-checks a new computation against an
old or deprecated one — look for comments like "sanity check", "TODO remove later", `old_*` variable
names — do not let it go as incidental cleanup. Such checks are left in deliberately to catch exactly
the divergence a refactor might introduce; removing one (typically because it references a deprecated
symbol) without independently confirming the two computations are equivalent for every supported type
risks reintroducing the bug it was guarding against, especially for library-extended types (custom
floating-point-like or integer-like types) that generic `<type_traits>` predicates don't recognize.

## correctness.stale-refs-after-rename (important, renames, moves, or splits of files, symbols, or modules anywhere in the repo)

<!-- provenance:
  #3177→#3192 cuda.parallel module split left docs automodule pointing at emptied package;
  #10012→#10042 docs flattening left stale path in a test comment and an empty api/thread toctree stub;
  #1075→#1108,#1110 lit.cfg path not updated after symlink removal broke local lit runs;
  #4537→#6516 NVTX macro rename left a stale #define in test_nvtx_disabled.cu, silently defanging a negative test;
  #4795→#4814 cudax detail→__detail rename swept ~120 files but missed the second example copy at top-level examples/cudax/
-->

When a diff renames, moves, or splits a file, macro, symbol, or module across many call sites, search
for surviving references to the old name/path — against the diff's base revision, not the current
checkout (a later fix may make an empty grep look safe). The diff will not show untouched files that
still point at the old location; enumerate them by purpose:
- Sphinx `automodule::`/`toctree::` directives, cross-references in `.rst`/`.md`, and code/test
  comments citing a doc or source path; verify every added `toctree::` entry leads to a page with
  actual content (auto-generated category pages can be empty stubs);
- test files referencing the name out-of-band: filenames containing `disabled`, `guard`, `negative`,
  `_neg`; tests that `#define` a library macro before including library headers (a negative test
  `#define`-ing the old name to `static_assert(false, …)` becomes a permanent no-op once real code
  stops using that name); `.in`/template files with hardcoded paths;
- sample/demo code in BOTH `<subproject>/examples/` AND top-level `examples/<subproject>/` — these are
  independently maintained, and a rename scoped to the subproject dir will not touch the latter.
For each candidate, state whether the rename covers it. Flag the omission even if the current checkout
looks consistent.

## correctness.cuda-runtime-symbol-version-guard (critical, code calling CUDA Runtime/Driver API symbols)

<!-- provenance:
  #2192→#5971 cudaGetDriverEntryPointByVersion gated only by a build-time CUDART_VERSION check, breaking when built against CTK>=12.5 but run against an older CUDA runtime (pair auto-inferred from issue #5970);
  #5976→#6895 (backport #6896) cuGetProcAddress switch reintroduced the same break by bootstrapping via the unversioned cudaGetDriverEntryPoint
-->

When a diff gates a call to a CUDA Runtime/Driver API symbol introduced in a specific CUDA Toolkit
version behind a build-time-only check (`_CCCL_CTK_AT_LEAST(...)`, `CUDART_VERSION >= ...`), flag it
unless the code also verifies the *linked/runtime* CUDA version before making the call
(`cudaRuntimeGetVersion`/`cudaDriverGetVersion`). CCCL is commonly built against a newer CTK than the
one on the machine that later runs the binary; a build-time-only guard links fine but hits an
"undefined symbol" failure at run time against an older `libcudart.so`. PR CI builds and runs against
the same CTK, so this is invisible to CI and only reproduces in the field.

## correctness.workaround-breaks-constexpr (important, constexpr-marked functions in cuda::std)

<!-- provenance:
  #5939→#7059 auto __tmp = mapping();
  return __tmp.is_exhaustive();
  workaround for a clang [[nodiscard]] warning made mdspan's is_exhaustive unusable in constexpr context on NVHPC;
  propagated to is_unique/is_strided/stride in #6703
-->

When a diff introduces a workaround inside a `constexpr` function (a named local copy of a temporary
to dodge a compiler warning, an intermediate variable, a cast), verify the function is still usable in
a constexpr evaluation context, not merely that it compiles as a runtime call. Some compilers (notably
NVHPC) diagnose "attempt to access run-time storage" for a materialized local copy of a class-type
temporary inside a constexpr call chain; the break only surfaces when a caller uses the function in a
`static_assert`, which the introducing PR's tests may not exercise. Prefer binding a `const auto&`
over a named copy, and require a `static_assert`-based test when adding such a workaround. If the
pattern is copy-pasted to sibling member functions, check each one.

## correctness.ptx-asm-operand-index (critical, hand-written/generated inline PTX `asm volatile` blocks)

<!-- provenance:
  #3440→#8403 128-bit atomic CAS codegen template misindexed asm operands (mov.b128 read from output/undefined and cross-mixed compare/desired registers), returning success while writing garbled data (intro corrected from issue #8402)
-->

When a diff adds or edits an inline `asm volatile("...", : outputs : inputs : clobbers)` block
referencing operands by number (`%0`, `%1`, …), manually verify each `%N` against its declared position
(outputs first, then inputs, in constraint-list order) — the compiler only checks that `%N` is in
range, not that it refers to the intended operand. Watch for: reading an output-only (`"="`) operand
instead of the matching input (reads undefined data), and adjacent operand groups that off-by-one after
a reorder (compare vs. desired in a CAS). Require a test that reads back the actual written value, not
just a success boolean — especially for >64-bit operand groups spanning multiple registers.

## correctness.partial-optimizer-workaround (important, compiler/optimizer-bug workarounds in CUDA kernels)

<!-- provenance:
  #8656→#8839 asm-volatile barrier for an NVCC codegen bug applied to one masked-copy pointer group in squadStoreBulkSync but not the structurally identical doStartCopy block below, causing an illegal memory access on sm_121 (intro corrected from issue #8838)
-->

When a diff adds a compiler/optimizer-bug workaround (an `asm volatile("" : "+r"(x))` barrier,
`#pragma`, `volatile`) to guard specific values against a cited codegen bug, search the surrounding
function and structurally similar siblings for OTHER occurrences deriving/consuming a value the same
way (a masked store from the same shifted-pointer arithmetic, a similar loop-carried value) and check
whether they need the identical barrier. A workaround applied to one of several structurally-identical
occurrences fixes the reported symptom but leaves the siblings silently exposed to the same miscompile.

## correctness.temp-storage-raw-alignment (critical, CUB/Thrust device-dispatch code allocating or indexing into `d_temp_storage`)

<!-- provenance:
  #6811→#9565 (backport #9781) warpspeed DeviceScan dispatch performed raw uint4 stores into d_temp_storage at a hand-computed offset, guarded only by a debug-only _CCCL_ASSERT and a "we probably need to ensure alignment" TODO, instead of routing through detail::alias_temporaries();
  misaligned callers hit Warp Misaligned Address faults on Blackwell (issue #9742)
-->

When a diff computes an offset into the caller-supplied `d_temp_storage` by hand
(`static_cast<char*>(d_temp_storage) + offset`, or passing it straight into a kernel argument) instead
of routing through the temp-storage aliasing utility (`detail::alias_temporaries`) that sibling
dispatch entry points already use, flag it — especially when the region is used for a wide or
vectorized load/store (`uint4`, or any type whose `alignof` exceeds 4/8 bytes). Only the aliasing
utility rounds the base pointer up and reports the required size; a hand-rolled offset inherits
whatever alignment the caller's allocation had. A `_CCCL_ASSERT(is_aligned(...))` is not sufficient —
it compiles out in release builds. A comment admitting uncertainty ("we probably need to ensure
alignment") next to such an assert means require the fix now, not as a follow-up.

## correctness.raii-move-no-disarm (critical, RAII/resource-owning/scope-guard types with move construction)

<!-- provenance:
  #5975→#10565 cudax scope_exit's move constructor was = default, copying the active flag without deactivating the moved-from source;
  both objects ran the cleanup action on destruction
-->

When a diff defines or defaults a move constructor for a type that runs an action or releases a
resource on destruction gated by an "active"/"engaged"/"owns" flag (or a handle nulled on release),
verify the move constructor disarms the moved-from source — the flag reset or handle nulled on the
object being moved out of, not merely copied. A defaulted move constructor is a strong signal: it
member-wise copies the active flag, so both source and destination stay engaged and both destructors
fire the cleanup. Applies to scope guards (`scope_exit` and friends), unique-handle wrappers, and any
RAII type exposing `release()`/`dismiss()`. Require a test that moves the guard and confirms the action
fires exactly once.

## correctness.hidden-friend-broad-constraint (important, hidden-friend operator/function templates defined inline in a class template)

<!-- provenance:
  #10687→#10723 constant_wrapper's hidden-friend operators, constrained only by a broad duck-typed trait after a narrower documented-workaround constraint was removed, were picked up by nvcc on Windows for unrelated classes via ADL, breaking main
-->

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

When a diff adds a type trait or overload constrained via SFINAE (`enable_if_t<…>`, a partial
specialization's template-argument expression, a `requires` clause) referencing a dependent type's
static member, verify the member is used the way that type's API defines it — a function (`E::rank()`)
vs. a value. A missing `()` raises no compile error inside a SFINAE context: the substitution becomes
ill-formed and the specialization silently drops out of matching instead of erroring. Mixing
`E::rank == 1` with `E::rank_dynamic() == 0` in one expression is a red flag. Also check the new path
is exercised by tests gated at the same (or looser) feature threshold as the implementation — a test
behind `#if _CCCL_STD_VER >= 2023` for code gated at `>= 2017` leaves the range in between untested.

## correctness.noexcept-wraps-throwing-impl (critical, new or changed async/fallible public member functions)

<!-- provenance:
  #7705→#10888 fixed_capacity_map's *_async members were noexcept while __open_addressing_impl ("@throws cuda_error") used _CCCL_TRY_CUDA_API;
  the cooperative-group launch branches also had no error check at all, unlike their cg_size==1 siblings
-->

When a diff adds or keeps `noexcept` on a public wrapper whose implementation documents or contains a
throwing failure path (`@throws` in the doc comment, `_CCCL_THROW`, a helper documented to throw), flag
the mismatch: an exception escaping a `noexcept` function calls `std::terminate`, turning a recoverable
CUDA launch error into a process crash, and it compiles cleanly with no warning. Especially likely when
error reporting is added at some but not all internal call sites, or in a lower layer without updating
the public exception specification. Also check that functions launching kernels along independent code
paths (fast path vs. cooperative-group path) follow EVERY launch with the same error check
(`_CCCL_TRY_CUDA_API`/`cudaGetLastError`) that sibling paths use.

## api.vocabulary-type-refactor (critical, public types in thrust/libcudacxx/cub)

<!-- provenance:
  #262→#1249 (backport #1292) pair trivial copyability;
  #454→#1286,#1425,#1497 complex reverted three times;
  #6393→#6403 variant modularization dropped monostate include from the umbrella header;
  #5839→#5925 swapping thrust::counting_iterator for cuda::counting_iterator lost any_system_tag under transform_iterator wrapping, selecting the wrong backend (issue #5860);
  reverted
-->

When a public vocabulary type (`thrust::pair`/`tuple`/`complex`, …) is reimplemented,
re-derived, or aliased, verify preservation of: trivial copyability and layout (downstream
`memcpy`s them), size/alignment/ABI, implicit conversions and promotions, overload
resolution, numerical behavior. rmm/PyTorch/RAPIDS break first — confirm the third-party
smoke tests ran (commit tags can skip them). The same applies when swapping a Thrust fancy iterator
for a `cuda::` counterpart across many call sites: check that its iterator category/system tag still
propagates when wrapped by ANOTHER fancy iterator (`thrust::transform_iterator`, `zip_iterator`) — a
category passed through unchanged instead of recomputed can silently downgrade `any_system_tag` to a
concrete system, selecting the wrong backend layers away from the swap. Require a test exercising the
new iterator through at least one other fancy-iterator wrapper, not only standalone.

## api.stricter-constraints-on-port (important, algorithm ports/reimplementations)

<!-- provenance:
  #1817→#2075 cub::DeviceMerge static_assert requiring identical value_type across both merge inputs, stricter than the thrust::merge implementation it replaced
-->

When an existing public algorithm is ported or reimplemented onto a new backend/dispatch layer
(e.g. a `thrust::` algorithm moved onto a `cub::Device...` implementation), flag newly added
`static_assert`/`enable_if`/trait constraints that are strictly narrower than what the previous
implementation accepted (e.g. requiring identical `value_type` across two input ranges when the old
implementation tolerated differing-but-related types). Confirm the narrowing is an intentional,
documented behavior change, not an incidental tightening — downstream consumers (rmm/cuDF/RAPIDS)
rely on the looser behavior.

## api.namespace-shadowing (important, public headers introducing new types/aliases)

<!-- provenance:
  #2343→#2372 cuda::experimental::stream_ref shadowed ::cuda::stream_ref, breaking unqualified uses in cudax/samples/vector_add (a file not touched by #2343)
-->

Flag a new type or alias introduced into a nested/inner namespace whose unqualified name collides with
an existing type from an enclosing namespace (e.g. `namespace cuda::experimental { struct stream_ref :
::cuda::stream_ref {…}; }` shadowing `::cuda::stream_ref`). Code that used the unqualified name from
within the inner namespace — including samples, tests, and downstream code — silently rebinds to the
new type. Grep the repo (including examples/samples) for unqualified uses of the shadowed name in
scopes affected by the new declaration; require qualification or confirm no ambiguity results.

## api.new-algorithm-missing-conventions (important, new CUB/Thrust device-algorithm entry points)

<!-- provenance:
  #2234→#4888 new cub::DeviceReduce RFA (deterministic reduce) kernel accepted a generic OffsetT via choose_offset_t but truncated it to int internally for grid sizing/indexing, and failed to unwrap fancy/proxy iterators before device use
-->

When a diff adds a new CUB/Thrust device-algorithm dispatch (`Dispatch*`/`*Kernel`), check it follows
the conventions of sibling algorithms: (1) if it accepts a generic `OffsetT`, every internal
kernel/component must support the full width (including 64-bit for >2^31-element problems) — flag any
`static_cast<int>(num_items)` or `int`-typed loop indices fed by an `OffsetT`-templated entry point,
unless a `static_assert` explicitly constrains `OffsetT` to what is supported; (2) input/output
iterators must be unwrapped via the standard fancy-iterator unwrapping utilities before being handed
to device code. New algorithms that skip these steps compile and pass small tests but silently
misbehave for larger inputs or common iterator types.

## api.alias-template-ctad-gap (important, public type aliases wrapping a class template that supports CTAD)

<!-- provenance:
  #3686→#6093 host_mdspan/device_mdspan/managed_mdspan alias templates over cuda::std::mdspan with a substituted accessor;
  CTAD silently failed to compile (pair auto-inferred from issue #6076)
-->

When a diff introduces a new public vocabulary type as an alias template
(`template <...> using NewName = Base<..., SubstitutedArg>;`) over a class template that supports
CTAD, verify CTAD still works for the new alias: the base's deduction guides deduce the *base's*
parameter list, which no longer matches once an accessor/policy/traits argument is substituted.
Explicit template-argument construction compiles, but `NewName x(args...);` fails — which the PR's own
tests often don't exercise. Either add deduction guides, implement the type as a proper class, or add
a compile-time test exercising CTAD for the new type.

## api.cross-project-name-collision (important, new or renamed public names in cuda/cub/thrust namespaces)

<!-- provenance:
  #5153→#6927 (backport #6937) cudax launch-argument utility renamed __launch_transform→device_transform, colliding with the cub::DeviceTransform algorithm family;
  renamed back after release branching (pair auto-inferred as #6927→#6937)
-->

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

## api.generic-param-replaces-concrete-type (important, public CUB/Thrust API overloads accepting a stream/config parameter)

<!-- provenance:
  #6204→#7915 (backport #8011) DeviceTransform Env=env<> default replaced the cudaStream_t stream parameter, breaking non-copyable types implicitly convertible to cudaStream_t
-->

When a diff replaces a concrete parameter type (e.g. `cudaStream_t stream = nullptr`) with a generic
templated parameter carrying a default (`Env env = {}`), check every call-site convention the old
parameter supported: arguments only implicitly convertible to the old type (a user stream wrapper with
`operator cudaStream_t()`) and non-copyable argument types. A pass-by-value template parameter binds to
the ARGUMENT's own deduced type, not the old parameter's type, so a non-copyable stream-like type that
used to convert-then-copy now fails to compile. Require an explicit non-template overload, or a test
passing a type that only converts to (rather than equals) the old parameter type.

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

## abi.internal-symbol-exposure (critical, new implementation-detail types/functions in cub/thrust/libcudacxx public headers)

<!-- provenance:
  #2591→#3209 CUB launcher factories and kernel-source getters added outside detail and without _CCCL_HIDE_FROM_ABI (pair auto-inferred from issue #2448);
  #5255→#5272 driver_api.h promoted from cudax into public libcudacxx tree carrying plain-inline functions without _CCCL_HOST_API
-->

When a diff adds internal-only plumbing to a public header (dispatch-layer launcher/factory,
kernel-source wrapper, policy wrapper, …), flag it unless: (1) the new type lives in a `detail` (or
anonymous) namespace — "not documented" does not remove it from the exported symbol surface; and
(2) any function whose result can differ between two copies of the library linked into one binary
(compiled against different CUDA toolkits), especially one returning a kernel or function pointer, is
marked `_CCCL_HIDE_FROM_ABI`. Otherwise, when two library copies end up in one program, the linker
silently keeps one definition and the losing copy's callers get the wrong kernel pointer — no build
error, just wrong launches. Candidate for a pre-commit grep: new `struct`/`class` directly under
`CUB_NAMESPACE_BEGIN`/`THRUST_NAMESPACE_BEGIN` outside a nested `detail`.

## perf.launch-attribute-consistency (important, multi-kernel CUDA dispatch changes)

<!-- provenance:
  #3114→#3199 PDL enabled at Partition/Merge triple_chevron launches but not the sibling BlockSort launch
-->

When a diff adds a per-launch enablement (e.g. a programmatic dependent launch attribute) alongside
kernel-body changes that assume it (`_CCCL_PDL_GRID_DEPENDENCY_SYNC`/`_CCCL_PDL_TRIGGER_NEXT_LAUNCH`),
do not just check the touched hunks: open the full `DispatchXxx::Invoke` (or equivalent) and enumerate
EVERY kernel launch it makes. Count sibling kernels whose bodies got the sync/trigger macro versus
launch sites in the same function that got the enabling attribute — a dispatch with 3 kernels where
only 2 launch sites changed is a red flag: the third is silently launched without the attribute,
leaving the optimization inert (no crash, no wrong result, just no perf gain).

## perf.benchmark-exec-tag-sync-without-sync-call (important, nvbench benchmark harness `state.exec(...)` calls)

<!-- provenance:
  #3114→#5350 merge_sort keys benchmark switched no_batch→sync while adding PDL although the exec lambda never synchronizes;
  reverted as an unnecessary workaround
-->

When a diff adds or changes an nvbench `exec_tag` on a `state.exec(...)` call to include
`nvbench::exec_tag::sync` (which tells nvbench "the KernelGenerator will perform CUDA synchronization
itself"), verify the lambda body actually performs an explicit synchronization
(`launch.get_stream().sync()`, `cudaStreamSynchronize`). Without one, nvbench will not synchronize on
its own and the measured time silently excludes some or all of the kernel's execution — invalid
numbers that can hide real regressions or manufacture fake speedups, especially for latency-hiding
features like programmatic dependent launch. If the lambda does not sync, `exec_tag::no_batch` or
`exec_tag::timer` is likely what was intended.

## perf.unbenchmarked-attribute-sweep (important, CUB/Thrust kernel signatures across dispatch/kernel files)

<!-- provenance:
  #6642→#9795,#9816,#9829,#9832,#9852,#9854,#9858,#9864 _CCCL_GRID_CONSTANT const added to nearly every CUB kernel parameter in one mechanical sweep, triggering a suspected NVCC/ptxas codegen bug (NVBug 6448961) that regressed DeviceSelect, radix sort, merge sort, ReduceByKey, and scan;
  took a half-dozen follow-up PRs culminating in a near-total revert
-->

When a diff adds (or removes) a compiler/codegen attribute — `_CCCL_GRID_CONSTANT`, `__restrict__`,
`__launch_bounds__`, a PDL attribute — on kernel parameter lists across many kernels/dispatch files in
one mechanical sweep ("add to every applicable kernel"), flag it unless the PR shows per-kernel
benchmark data for each affected algorithm family, not just a compile check or one representative
kernel. These attributes change code generation (parameters in constant memory, register allocation)
and their effect is kernel-, type-, and architecture-dependent; a win for one kernel can silently
regress a sibling with no compile error — only a benchmark sweep catches it. Treat "applied uniformly
to all applicable kernels" as a red flag, and prefer an incremental, benchmark-gated rollout per
algorithm family over a blanket repo-wide application.

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

## perf.functor-property-proclamation-in-port (important, thrust/cub backend glue dispatching user functors)

<!-- provenance:
  #6012→#9071 tabulate port to transform_n wrapped the user functor in proclaim_copyable_arguments, forcing the copied-arguments dispatch path and slowing wide row-wise workloads ~3.5x (issue #9070)
-->

When a diff routes an algorithm through a shared backend (e.g. porting a thrust algorithm onto
`transform`) and wraps the user's functor in a property-proclaiming adapter
(`proclaim_copyable_arguments`, `proclaim_return_type`, address-stability markers), flag it: such
wrappers change which dispatch path the backend selects, and the forced path can be several times
slower for some workload shapes. Require benchmarks against the pre-port implementation across
workload shapes, and question whether the proclaimed property can even be asserted for arbitrary user
functors.

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

## test.remove-negative-test-without-replacement (important, test suites in cub/thrust/libcudacxx)

<!-- provenance:
  #3970→#9211 (issue #807) generate/raw_reference_cast simplification deleted runtime_static_assert.h, unittest_static_assert.cu, and generate_const_iterators.cu — the compile-fail harness — with no replacement until #9211 added *_fail.cu tests
-->

When a diff deletes a test file, header, or CMake target whose purpose is to verify a negative
property (a static assertion fires, a call fails to compile, an operation is rejected —
filenames/macros like `*_fail*`, `*_static_assert*`, `UNSUPPORTED`/`XFAIL` markers), check that an
equivalent replacement lands in the same diff, even when the deletion is incidental to an unrelated
refactor. "The old mechanism was awkward/incompatible with a compiler" is not sufficient justification
for deleting it outright — the property it verified still needs coverage. Also check whether any CMake
test-registration still references the deleted test name.

## infra.pin-deps (important, CMake/CI/submodules)

<!-- provenance:
  #534 nvbench `#main` →#582
-->

Flag dependencies fetched by branch name (`CPMAddPackage("gh:org/repo#main")`, `GIT_TAG
main`); pin a commit or tag. Candidate for a pre-commit grep.

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

## infra.bulk-regen-drops-manual-entry (important, devcontainer/CI generated-file regeneration diffs)

<!-- provenance:
  #1935→#1955 CUDA 12.5 devcontainer bump deleted the hand-maintained RAPIDS devcontainer symlink with no replacement, breaking RAPIDS CI
-->

A diff that bulk-regenerates a templated file set (e.g. bumping the CUDA version across every
`.devcontainer/cudaX.Y-<compiler>/devcontainer.json`) shows a uniform pattern: each deleted file has a
regenerated counterpart. Flag any deleted entry that does NOT fit the pattern — a symlink, a
name/version not part of the bump, or a deletion with no corresponding addition. Such outliers are
usually hand-maintained special cases the regeneration script doesn't know about; grep the deleted
name across `.github/workflows/`, `ci/`, and `.devcontainer/` before approving the removal.

## infra.version-bump-behind-ci-skip (important, pre-commit hook / pinned CI-tool version bumps)

<!-- provenance:
  #10729→#10876 mirrors-mypy bumped v1.16.1→v2.1.0 while mypy sat on the pre-commit.ci skip: list, so green CI never exercised the new version;
  it then crashed locally on a config older mypy tolerated
-->

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

When a diff removes, renames, or deprecates (e.g. `_CCCL_DEPRECATED`) a public macro, function, class,
or type, verify that `docs/cccl/*_migration_guide.rst` gains a matching entry describing the removal
and its replacement (or "No replacement"). Applies equally to mainline PRs and backport PRs. Flag a
diff that deletes/deprecates public symbols across `.cuh`/`.h` files without any corresponding
migration-guide update.

## docs.link-target-topic (important, diffs changing documentation URLs in docs or code comments)

<!-- provenance:
  #10887→#10895 bulk CUDA-guide link migration pointed memcpy_async performance guidance at the device-callable-APIs appendix instead of the async-copies page
-->

When a diff replaces an external documentation URL, do not stop at "the new URL resolves": check that
the new page/section covers the same topic that the surrounding prose and the old link's anchor
promise. Anchor and path names carry intent — flag a replacement where the old anchor's topic (e.g.
performance guidance, tuning, semantics) is swapped for a generic anchor on a differently-themed page.
Be especially suspicious in bulk link migrations where one old→new mapping was applied mechanically to
every occurrence; each occurrence needs its own topical match.

## Retired

<!-- provenance: #651, #1257 -->

- `if constexpr` fallback instantiation firing `static_assert`s pre-C++17: obsolete since
  C++17 is the minimum dialect.
