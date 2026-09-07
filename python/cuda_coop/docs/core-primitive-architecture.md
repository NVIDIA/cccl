# Backend-neutral cooperative primitive semantics

`cuda.coop._core` is the internal, backend-free description layer for shared
cooperative primitive semantics. It uses the Python standard library and
does not import numba-cuda-mlir, CUTLASS, compiler
extensions, linkers, caches, launchers, or provider runtimes.

The dependency direction is one-way:

```text
public backend frontend
    -> frontend validation and normalization
    -> cuda.coop._core semantic builder
    -> backend adapter
    -> backend compiler / rewrite / provider / cache / launcher
```

Core records the primitive specialization. Backend code owns compilation,
linking, caching, launching, and integration with tracing or rewrite systems.

## Core model

The core layer provides immutable records for:

- template parameters, dependencies, constants, and runtime-value markers;
- scalar values, pointers, pointer-relative element offsets, references,
  fixed-size arrays, and temporary storage;
- C++ operators, stateless Python operators, and stateful Python operators;
- unspecialized `Algorithm` descriptions and immutable `AlgorithmSpec`
  specializations;
- CUDA `ThreadHierarchy` and `ThreadGroup` descriptors, including
  planner-resolved static extents and current-group identity;
- stable semantic identities and symbol-mangling inputs; and
- the `CoreBackendAdapter` lowering protocol.

`ParameterRole` records semantic input, output, in/out, temporary-storage,
constant, operator, and state roles without tying them to a backend ABI.
`is_return` separates semantic output intent from backend return behavior. An
output array sets `is_output=True` and `is_return=False` because the caller
provides its storage. A scalar output may use a backend-generated result slot.

An `AlgorithmSpec` freezes its specialization mapping and caches its semantic
identity. Declared C++ template arguments retain declaration order. Auxiliary
specialization values, such as scan `ITEMS_PER_THREAD`, follow in sorted order
for deterministic identity and lowering without becoming extra C++ class
template arguments.

Semantic identities record runtime/static classification and omit runtime
operand values. Callable identities include code, defaults, closures,
partial arguments, and callable-object state so distinct operators cannot
produce an invalid shared provider or compiler cache entry.

## Compatibility notes during core migration

The core migration preserves supported public primitive names, call shapes,
generated symbol components, and backend-specific lowering behavior. It also
deliberately rejects inputs that older frontend-local builders accidentally
accepted while constructing invalid or ambiguous CUB specializations. The
current consolidated behavior-change list is:

- boolean or fractional item counts are no longer coerced with `int(...)`;
- CuTe run-length count options accept non-boolean `numbers.Integral` values,
  including NumPy integer scalars, under the shared positive-integer contract;
- CuTe row-reduce geometry now follows that same integral contract rather than
  accepting native `int` objects only;
- row-reduce shapes exceeding one 1024-thread CUDA block now fail before C++
  generation instead of reaching CUB's class-level `static_assert`;
- row-reduce calls whose launch width differs from
  `rows_per_block * warps_per_row * 32` now fail during CuTe scratch planning
  instead of silently indexing the wrong row partials;
- CuTe row-reduce scratch planning now includes CUB's full-warp `NullType`
  storage prefix and alignment before the row-partial array; the previous
  `logical_warps * sizeof(T)` estimate under-allocated some multi-warp rows.
  The common float32, one-row, four-warp requirement grows from 16 to 20 bytes;
  callers with a hard-coded 16-byte buffer must enlarge it and now receive a
  clear host-side error instead of running with shared-memory out-of-bounds
  undefined behavior;
- truthy non-boolean algorithm or layout flags are no longer accepted as
  booleans;
- paired options such as radix bounds must be present together and satisfy
  their static range constraints; and
- stateful constructor operand sets must describe one complete supported CUB
  constructor rather than a partial parameter list.

These cases now fail early with `ValueError`. This section is the consolidated
index while the backend-neutral core series remains under development; the
individual primitive slices below are authoritative about which checks apply
to each API. A package release that includes the series should copy this index
into its user-facing release notes. The shared validators also replace some
backend-specific diagnostic strings with messages about positive integers and
complete operand sets; callers must not rely on the legacy message text.

The active frontends keep their established call shapes while `_core` removes
accidental semantic drift. Numba-CUDA-MLIR retains full and partial Load/Store
signatures, its pointer-offset form, optional in-place and out-of-place
Exchange signatures, and both default- and explicit-bit Radix Sort overloads.
Its fused Run Length Decode path may select the default-window overload. CuTe
continues to use explicit bit ranges and window offsets where its public
surface requires them. The primitive slices below define the exact conditions
and backend ownership.

## Backend adapters

Adapters implement dtype normalization, C++ type rendering, parameter and
operator lowering, temporary-storage treatment, and final materialization.
The shared `lower_method_parameters` dispatcher selects the appropriate hook;
the backend decides which concrete compiler objects those hooks create.

The Numba-CUDA-MLIR adapter translates core records into its existing
`Algorithm` and parameter types. It leaves LTOIR/PTX compilation, unique symbol
IDs, source generation, linker inputs, single-phase rewriting, and invocation
objects in the backend modules. The CuTe TopK, BlockLoad, BlockStore,
BlockShuffle, BlockRowReduce, BlockDiscontinuity, BlockHistogram,
BlockRunLengthDecode, logical-warp Reduce, logical/partial/`ThreadData`
WarpScan, and logical/scatter WarpExchange providers derive their compatibility
requests or indexing plans from the same core semantics. Full block and
physical-warp CuTe Reduce, Scan, Exchange, and MergeSort, plus block RadixRank
and RadixSort, instead consume shared group plans and materialize official
CUDAX/CUB artifacts. The private logical-warp MergeSort adapter consumes that
same plan and public-CUB artifact with one storage instance per logical warp.
The private BlockExchange adapter's six extended modes use the same whole-array
public-CUB provider as group Exchange, but build a direct block specialization.

## CUTLASS public-provider validation

The private CUTLASS capability inventory records implementation readiness for
canonical public CUDAX/CUB routes. Planner, runtime, and final-cubin tests are
the stable acceptance surfaces. `READY` does not imply identical machine code
for every dtype, launch shape, compiler, or architecture.

Optional scripts under `tools/run_cutlass_*_validation.py` compare representative
candidate kernels with direct CUB and retain numerical, source-contract,
normalized-SASS, resource, function-attribute, and occupancy diagnostics. Their
route matrices may evolve with the investigation and do not determine
readiness.

There are no current `BLOCKED_PROVIDER_PARITY` records. Row Reduce remains
`BLOCKED_PLANNER_AND_DEPENDENCY`. Key and key/value group-first TopK are `READY` through
the exact pinned CUB-detail provider used by qualified pair TopK.

## BlockTopK validation slice

`cuda.coop._core.block.make_block_topk_spec` captures:

- key-only versus key/value payloads;
- max versus min selection;
- block dimensions and items per thread;
- full-tile versus partial-tile policy;
- omitted, static, and runtime `num_valid` and radix-bit arguments; and
- semantic temporary storage plus in/out key and value arrays.

One canonical `cub::BlockTopKCoop` alias supplies named forwarding methods for
all eight selection, payload, and tile-policy combinations. Numba-CUDA-MLIR
materializes that alias through its adapter. CuTe maps the same normalized
selection and specialization shape into its provider request.

TopK tests cover dimensions, arrays, static/runtime options, C++ type aliases,
temporary storage, and a provider-style backend without introducing operator
compilation.

## BlockScan operator and callback gate

`cuda.coop._core.block.make_block_scan_spec` owns the shared BlockScan
signature matrix:

- inclusive versus exclusive mode;
- scalar-return versus caller-provided array output;
- sum versus explicit scan operator methods;
- optional initial values;
- C++ functors, stateless Python operators, and stateful Python operators;
- stateful or stateless prefix callbacks;
- optional block-aggregate output pointers; and
- CUB algorithm and multidimensional block specialization values.

Frontend-specific validation remains outside core. Numba-CUDA-MLIR retains
its initial-value defaults, factory constants, scalar/array selection rules,
and alias validation. After normalization, it lowers the shared signature
through its adapter.

Scan tests show that core classifies operator definitions as static and
stateful callback storage as a runtime operand. Backend code compiles Python
functions, chooses PTX or LTOIR, creates ABI wrappers, and integrates prefix
state with its rewrite or invocation path.

CUTLASS CuTe root and private block-adapter Scan calls converge on one
group-lowering plan and canonical provider that calls public `cub::BlockScan`.
The root block surface supports scalar and `ThreadData` operands plus the
`raking`, `raking_memoize`, and `warp_scans` algorithms. Root physical-warp Scan
and the overlapping full-warp scalar private-adapter calls likewise share one
canonical public `cub::WarpScan` provider. Private logical-warp and valid-item
scalar forms remain direct-CUB compatibility routes through the legacy
provider; private `ThreadData` forms retain its generated per-lane folding
adapter. None of these compatibility forms expands the root group-first operand
surface.

Block Scan's old handwritten implementation is intentionally not a runtime or
test fallback: an automatic non-CUB collective would violate the provider
contract. Direct CUB remains an optional comparison oracle. Planner, runtime,
and final-cubin tests establish readiness for block and physical-warp Scan
without extending that conclusion to every dtype or architecture.

## BlockReduce call and specialization slice

`cuda.coop._core.block.make_block_reduce_semantics` describes the
dimension-independent CUB call contract: custom `Reduce` versus optimized
`Sum`, scalar versus blocked-array input, the reduction operator, optional
scalar valid-thread count, temporary storage, and the synthetic scalar output
used by backend wrappers. `make_block_reduce_spec` adds the static block shape
and CUB algorithm needed to materialize `cub::BlockReduce`.

Numba-CUDA-MLIR builds the full specialization and lowers it through its
adapter. CUTLASS CuTe root and private-adapter Reduce calls converge
on one group-lowering plan and provider artifact. Full block and physical-warp
defaults call broadcasted `cuda::experimental::coop::reduce`; supplied scalar
counts and explicit block algorithms call the exact `cub::BlockReduce` or
`cub::WarpReduce` specialization. Logical subwarps remain a private
CUB adapter. Valid-count block and physical-warp `ThreadData` is rejected because
those direct-CUB routes have no partial-array overload; callers that need that
policy must explicitly fold each thread's items to a scalar first. The logical
subwarp adapter intentionally preserves its legacy exception: it folds each
lane's items inside the wrapper and returns a one-item `ThreadData` whose value
is defined only on logical lane zero.
Any supplied valid-count binding selects direct CUB and root-only visibility,
even when a static or runtime count equals the full group size. Callers that
want the broadcasted full-group CUDAX route must omit the count.

This convergence intentionally changes the private compatibility contract:
full physical block/warp `ThreadData` reductions return a scalar rather than a
one-item container, and the full result is broadcast rather than root-only.
Callers should remove `result[0]`; lane-zero-only consumption remains valid but
is no longer necessary. Valid-count direct-CUB results remain root-only, and every
group member must still invoke the collective in converged control flow.
Passing the former one-item result to private store helpers is also a migration:
write the returned scalar directly on every member for full broadcast routes,
or only on rank zero for root-owned routes.

Floating-point compatibility is semantic rather than bitwise across this
implementation change. The selected official CUDAX/CUB tree may reassociate
operations and therefore change rounding, NaN selection/propagation, or signed
zero relative to the retired provider. Root and private-adapter calls selecting the same
artifact must remain bitwise identical to each other; numerical correctness
uses dtype-appropriate tolerances and does not pin the retired tree. For one
route and toolchain, the official CUDAX/CUB implementation is normative for
NaN payload/selection, infinities, and signed zero; matching Python and direct
C++ probes must be bitwise identical for those cases. Unsigned integer
arithmetic follows the selected type's modular C++ semantics, while inputs that
would cause signed integer overflow are unsupported. No cross-route or
cross-algorithm bitwise equivalence is promised.

Core always records temporary storage as part of the collective's semantics.
An adapter may omit that runtime operand when its backend allocates implicit
storage, or lower it when the caller supplies explicit storage. CUTLASS CuTe
Reduce, Scan, and Exchange accept legacy block `TempStorage` syntax for
compatibility, but their shared plans use implementation-owned CUDAX/CUB scratch
and do not charge an explicitly sized compatibility object.

CUTLASS exposes the canonical root `coop.TempStorage`; an internal compatibility
alias preserves implementation identity. An omitted-size object switches
group-first block Load, Store, Scan, AdjacentDifference, Discontinuity,
RadixSort, MergeSort, and Exchange to caller-owned shared scratch. The CUB
wrapper receives fixed i32 address/size operands, validates them against the
exact compiled C++ type, and omits its implementation-owned static allocation.
Shared storage defaults to `auto_sync=True`, so the external-scratch helper
inserts a post-call CTA barrier before another collective reuses the object.
`auto_sync=False` selects manual `storage.sync()` calls. Exclusive call-site
slices do not alias, default to no barrier, and reject `auto_sync=True`.

The trace finalizer obtains `sizeof` and `alignof` from NVRTC name expressions
registered in the same single LTO-IR compilation as the cooperative shims. It
solves every kernel/storage plan before MLIR mutation, aliases every shared use
at offset zero with the strongest size and alignment, assigns each exclusive
call site an aligned disjoint slice, inserts one allocation per identity, and
backpatches the placeholders before module hashing. This analysis is
conservative and does not perform branch-sensitive liveness. Warp Scan and
providers without exact-layout registration remain implementation-owned.
For CUDAX, each generated specialization has implementation-owned static shared
scratch in the official header. Equal semantic requests deduplicate to one
provider artifact and reuse its scratch after the planner-required barrier;
distinct artifacts are not assumed to alias. Direct-CUB wrapper artifacts own
their own static shared storage. Native block/warp Scan aggregate output uses
the same CUB overload and `TempStorage` as the Scan itself; it adds an output
ABI, not a second reduction allocation. A separate Scan-plus-Reduce
composition does add independent CUDAX reduction scratch, so final-cubin
shared-memory and occupancy checks must cover that coexistence rather than
infer a peak from either source type.

Numba-CUDA-MLIR preserves the existing `BlockReduce`, `block_reduce`, and
`Sum`/`Reduce` symbol components. Its tests pin the materialized symbol base
and the core symbol-mangling inputs in addition to the parameter ABI.

## BlockRowReduce optional CuTe slice

`cuda.coop._core.block.make_block_row_reduce_spec` owns the static
`cub::BlockRowReduceWarpBroadcast<T, ROWS_PER_BLOCK, WARPS_PER_ROW>::Sum`
contract: row geometry, template order, include, temporary storage, scalar input,
and synthetic scalar output. This is distinct from general `BlockReduce`; its
rows and warps describe one statically partitioned CTA rather than an arbitrary
runtime-width reduction. Core models only `Sum` because that is the operation
retained by the private row-reduction adapter. There is no
public group-first row-sum operation while the required CUB header is absent
from the selected CCCL tree. The header's generic commutative reduction entry
point is not part of this slice; exposing it would be a separate, non-binding
future operation/operator design.

The required `cub/block/block_row_reduce.cuh` is an in-repository CUB dependency,
not a CUDA Toolkit dependency. `cuda.coop` intentionally evolves against the
CUB headers from the same CCCL source tree or from the co-installed
`cccl-headers` bundle built from that tree. CUDA Toolkit include directories
may still provide CUDA compiler and runtime headers, but they are never accepted
as the CUB source. Provider compilation validates and selects exactly one CUDA
Toolkit include directory by precedence; it never concatenates roots from
different Toolkit installations into one compilation. CCCL-only consumers do
not require a CUDA Toolkit to inspect the bundled support headers. Source
builds discover the enclosing CCCL checkout;
`CUDA_COOP_CCCL_ROOT` is the common explicit override, and
`CUDA_COOP_CUTLASS_PROVIDER_CCCL_ROOT` is the CUTLASS-specific override.
If the selected CCCL header set lacks an individual required header, provider
compilation fails before NVRTC rather than falling through to a toolkit copy.
The row-sum runtime tests likewise inspect the repository header and skip until
the corresponding CUB addition is present on the same branch.

CuTe's typed row-reduce request carries the core spec directly. Bundle include,
CUB class, method, and template argument order are rendered from that spec,
while the existing `cuda_coop_cutlass_row_sum_<dtype>_r<rows>_w<warps>` symbol,
shared-memory address/size/sync FFI ABI, scalar-only payload rule, scratch
planning, launch-fact resolution, provider sessions, compilation, and linking
remain backend-owned. Geometry is normalized by core before dtype resolution
to retain the direct provider's diagnostic ordering. During the public CuTe
path, scratch planning resolves authoritative compiler launch facts, including
the current kernel's exact `reqntid` attribute, and requires its width to equal
`rows_per_block * warps_per_row * 32`. Missing or mismatched launch information
fails before provider registration. Exact equality is required because CUB
derives the participating block width from the row geometry; surplus and
missing threads are both invalid. The registered CuTe provider preserves those
launch facts and repeats this check immediately before its terminal FFI
hook. That terminal hook cannot validate launch width itself because its ABI
receives only the lane value and static row geometry, so it is private and named
`_provider_row_sum_after_launch_validation` to make the precondition explicit.
No public root API invokes that terminal hook directly. The retained private
scoped `row_sum` compatibility adapter requires `TempStorage`; its single-phase
dispatcher validates the launch before it records that storage use, and the
registered provider then repeats the launch check. Specialized row reduction
remains absent from the group-first roots and blocked on
`cub/block/block_row_reduce.cuh`. Supported group-first code uses the declared
`reduce(group, ...)` routes, which do not call this terminal row-sum hook. The
pure `infer_temp_storage_requirement("row_sum", ...)` query depends only on row
geometry and dtype, so autotuners, static shared-memory budgeters, and
documentation generators can size a candidate without fabricating launch
metadata. Launch validation remains in the dispatch path where the metadata
naturally exists.

For `WarpsPerRow > 1`, the current CUB 3.5 warp-broadcast implementation lays
out one byte of full-warp `WarpReduce::TempStorage` (`NullType`) per logical
warp, followed by an aligned `Uninitialized<T>` partial per logical warp. This
is an internal-layout observation, not a stable CUB contract. CuTe uses it as a
best-effort mirror:
`align_up(logical_warps, sizeof(T)) + logical_warps * sizeof(T)` for its closed
set of built-in scalar types, whose size and alignment match. Single-warp rows
select CUB's one-byte empty storage. Core intentionally accepts any non-`None`
dtype because admissible scalar representations are backend concerns. CuTe's
current public type registry contains only 1-, 4-, and 8-byte scalars; its width
check is a forward guard, and adding another representation requires an
explicit layout review. The generated shim additionally compares the supplied
byte count with the compiled C++ `sizeof(TempStorage)`. If a future CUB layout
grows beyond the mirror, the compile-time `static_assert` stops every affected
provider build with an update-required diagnostic; it is a tripwire, not a
self-healing fallback. A second assertion checks that CuTe's planned alignment
covers `alignof(TempStorage)`. If CUB shrinks the type, CuTe remains correct but
temporarily over-allocates shared memory until the mirror is updated. The
runtime trap separately rejects a caller buffer smaller than CUB's actual type.
Supported public calls first get the normal
`TempStorage size_in_bytes is smaller` error, so the diagnostic-free device trap
is only a backstop for a bypassed host-side check. The GPU integration test
compiles this assertion when the optional CUB header is available; environments
without that header skip the runtime test and intentionally have no host-only
CUB layout cross-check. Likewise, pure host-only sizing consumers receive the
best-effort mirror without compiling either assertion; CUB drift is detected
only when an actual provider is built.

## BlockMergeSort payload and partial-tile slice

`cuda.coop._core.block.make_block_merge_sort_semantics` describes the
dimension-independent `BlockMergeSort::Sort` call contract: keys-only versus
key/value payloads, items per thread, in-place arrays, a strict-weak-ordering
comparator descriptor, and the full- versus partial-tile overload. Runtime
`valid_items` and `oob_default` values are not retained in core; their presence
selects the overload and semantic identity. `make_block_merge_sort_spec` adds
the static multidimensional block shape required by `cub::BlockMergeSort`.

Numba-CUDA-MLIR materializes that specialization through the shared adapter.
Its public normalization, user-defined-type policy, callable compilation,
rewrite, linker, and invocable behavior remain backend-owned.

CUTLASS root and private-adapter forms build one `GroupMergeSortSemantics` plan. Exact
launch dimensions specialize `cub::BlockMergeSort`; the total CTA thread count
must be a power of two. Physical-warp root calls specialize
`cub::WarpMergeSort` at width 32, while private complete logical-warp adapters select a
width-specific specialization and one `TempStorage` instance per logical warp.
`maxntid` is never accepted as an exact launch fact.

The reusable CUTLASS core adapter lowers typed in/out arrays, the static C++
less/greater operator, implementation-owned storage, and multi-value result
packing. A generated wrapper may only adapt arguments, make one public CUB
`Sort` call, unpack keys and optional values, and emit the required block- or
warp-scope storage-reuse barrier. Runtime `valid_items` and `oob_default`
select the partial block or warp overload but remain outside artifact identity.
There is no handwritten compare/exchange fallback. Equal-key ordering is
explicitly unstable; key/value association is preserved in the valid sorted
prefix, while partial-tile outputs beyond that prefix are unspecified.

Pure tests pin multidimensional shape, payload, tile, parameter-role, semantic
identity, exact storage ownership, and root/private-adapter artifact identity. Provider
tests pin deterministic symbols, one-call generated source, typed result ABIs,
logical-warp storage indexing, NVRTC compilation, final-link elimination, and
GPU keys/pairs behavior.

## BlockLoad and BlockStore data-movement slice

`cuda.coop._core.block.make_block_load_store_semantics` owns the
dimension-independent data-movement contract: load versus store, items per
thread, the CUB algorithm domain, full versus partial tiles, optional load
default values, pointer/array roles, and temporary storage. The corresponding
`make_block_load_spec` and `make_block_store_spec` builders add the static
multidimensional block shape required by CUB.

Numba-CUDA-MLIR retains the full and partial overloads together and adds an
element-offset operand that rewrites the preceding source or destination
pointer. It materializes those overloads from the core template and signature
graph, while enum handling, public aliases, compilation, and invocation remain
frontend-owned.

CuTe learns the block shape from the running kernel. Group-first calls and
eligible private-adapter calls with exact complete-block facts, a supported
dtype, and a raw compact pointer materialize the shared public-CUB plan.
Group-first calls support the CUB algorithm matrix; private adapters collapse
`vectorize` to `direct` and reject transpose algorithms before route selection.
A non-contiguous, statically unproven, or CUB-incompatible private tensor route
retains the explicit CuTe indexing payload adapter. Tensor tracing, adapter
indexing, predication, and element access remain CuTe-owned.

The existing `BlockLoad`, `BlockStore`, `block_load`, `block_store`, `Load`, and
`Store` symbol components are unchanged, so this migration does not require
artifact regeneration.

## BlockExchange mode and overload slice

`cuda.coop._core.block.make_block_exchange_semantics` owns the
dimension-independent `BlockExchange` contract across all eight CUB movement
modes. It records in-place, out-of-place, or combined overload forms; items per
thread; optional warp time slicing; scatter-rank and validity-flag dtypes; and
the exact temporary-storage and array parameter order. The corresponding
`make_block_exchange_spec` builder adds the static multidimensional block shape
and CUB class template arguments.

Numba-CUDA-MLIR retains both in-place and out-of-place signatures when
`use_output_items` is unspecified. It keeps enum normalization,
user-defined-type wrapping, compilation, rewriting, and invocation behavior in
its backend modules while materializing the core graph.

CuTe consumes the dimension-independent contract after resolving its
`ThreadData` value, rank, and flag types. Complete-block striped-to-blocked and
blocked-to-striped calls from the root or private adapter share one group plan
and whole-array public-`cub::BlockExchange` provider. The six additional block
modes remain private, but use the same provider renderer through a direct block
specialization. Generated artifacts own exact typed scratch; a private
`TempStorage` operand is accepted but ignored and uncharged. Runtime launch,
rank, and flag validation plus FFI result construction remain CuTe-owned.

Core uses the canonical `block_exchange` C name for every mode, matching the
shared WarpExchange pattern. This replaces Numba-CUDA-MLIR's older
mode-suffixed C names; generated MLIR symbols and compile artifacts keyed by
those names must be regenerated. These are internal device-wrapper symbols, so
source consumers need no API changes, but package builders and owners of
cached or prebuilt MLIR artifacts must rebuild them. Mode identity remains
part of the per-mode CUB `method_name`; Numba-CUDA-MLIR mangles that method
name together with the canonical C name and parameter types, keeping
same-shape modes distinct.

## BlockShuffle scalar, array, and boundary slice

`cuda.coop._core.block.make_block_shuffle_semantics` owns normalized offset,
rotate, up, and down modes; scalar versus per-thread-array shape; distance
binding; and optional block-prefix or block-suffix outputs. The CUB-specific
`make_block_shuffle_spec` layer accepts CUB's scalar Offset/Rotate methods and
array Up/Down methods, adds the static multidimensional block shape, and owns
the exact temporary-storage, input, output, distance, and boundary parameter
roles. CuTe-only multi-item Offset/Rotate and arbitrary-distance array forms
are intentionally outside the CUB spec and are rejected there; they remain
valid only in the broader dimension-independent contract.

The generic omitted/static/runtime `ArgumentBinding` records now live in
`cuda.coop._core` rather than under BlockTopK. TopK keeps its existing
`cuda.coop._core.block` imports, while BlockShuffle uses the same records to
describe Numba-CUDA-MLIR's compile-time distance without retaining runtime
payload values in semantic identity.

Numba-CUDA-MLIR keeps public enum normalization, scalar Up/Down conversion to
signed CUB Offset calls, UDT wrapping, rewriting, compilation, and invocation
construction in its adapter. CuTe consumes the broader
dimension-independent contract because its shims also support arbitrary
distance and multi-item offset/rotate forms; tracing, scratch planning, local
item selection, request registration, generated helpers, and FFI calls remain
CuTe-owned. Existing `block_shuffle` C names and CUB-valid method symbols are
unchanged. Numba-CUDA-MLIR's single-phase rewrite treats keyword
`block_prefix` and `block_suffix` arrays as runtime outputs (while explicit
`None` remains no boundary), matching its positional boundary overloads.

## BlockAdjacentDifference overload slice

`cuda.coop._core.block.make_block_adjacent_difference_semantics` owns left
versus right neighbor selection, full versus partial tiles, optional external
predecessor or successor items, items per thread, the binary difference
operator, and exact parameter order. Runtime valid-item and boundary values
select an overload without becoming part of semantic or cache identity.
`make_block_adjacent_difference_spec` adds the static multidimensional block
shape required by `cub::BlockAdjacentDifference`.

Numba-CUDA-MLIR retains its public enums, callable compilation, UDT wrapping,
single-phase rewrites, linking, and invocable construction while materializing
the core signature. CuTe uses the dimension-independent
contract to select its existing runtime-width shim; request registration,
generated C++, group-width discovery, shared storage, and FFI calls remain in
the provider. CuTe consumes named fields from the contract rather than
lowering the core parameter tuple as its positional FFI ABI. Its current
frontend accepts built-in subtraction only, so the provider records that fixed
operator in core while retaining callable rejection in the CuTe frontend. The
operator descriptor is semantic metadata for this parity check; CuTe's
generated helper continues to emit the fixed subtraction directly.

The core overload matrix records an important CUB asymmetry: left partial
tiles may accept a predecessor, but `SubtractRightPartialTile` has no overload
that accepts a successor. The active frontends reject that nonexistent call.
Existing
`BlockAdjacentDifference`, `block_adjacent_difference`, and public CUB method
symbol components remain unchanged. CUTLASS root and private block-adapter calls can
select the same exact-layout deferred scratch contract as the other registered
block families; default and explicitly sized legacy calls keep their
implementation-owned ABI and generated reuse barrier.

## BlockDiscontinuity output and boundary slice

`cuda.coop._core.block.make_block_discontinuity_semantics` owns head, tail,
and paired output modes; input and flag dtypes; items per thread; the binary
flag predicate; and the exact CUB overload ordering for optional external
predecessor and successor items. `make_block_discontinuity_spec` adds the
static multidimensional block shape. BlockAdjacentDifference and
BlockDiscontinuity share the core `BlockTileBoundary` vocabulary instead of
maintaining separate encodings for none, predecessor, successor, and both.
`BlockAdjacentDifferenceBoundary` remains an alias for compatibility; that
primitive's mutual-exclusion validation keeps the generic `BOTH` member
unreachable there. The shared alias does make `BOTH` visible through the
adjacent-difference enum name, although calls using both boundary directions
remain invalid.

Numba-CUDA-MLIR retains public enums, output-shape inference, callable
compilation, UDT wrapping, single-phase argument reordering, linking, and
invocation construction while materializing the core signature.
CuTe records its fixed not-equal predicate and `Int32` flag type in the
dimension-independent contract, then maps the normalized mode and item count
to its existing runtime-width requests. The operator descriptor and parameter
tuple are semantic parity metadata for CuTe; generated helpers continue to
emit their fixed comparison directly, and provider registration, group-width
discovery, shared storage, paired-request composition, and FFI calls remain
provider-owned.

The existing `BlockDiscontinuity`, `block_discontinuity`, `FlagHeads`,
`FlagTails`, and `FlagHeadsAndTails` symbol components are unchanged. Pure
tests pin all four paired-boundary overloads and semantic identity, and adapter
tests pin materialized ABIs. The Numba-CUDA-MLIR GPU suite exercises heads
with a predecessor and the paired predecessor/successor overload. CUTLASS root
and private block-adapter calls can select caller-owned deferred scratch with the same exact CUB
layout probes, shared/exclusive planner, and fixed address/size ABI used by
BlockAdjacentDifference.

## BlockHistogram lifecycle and algorithm slice

`cuda.coop._core.block.make_block_histogram_semantics` owns the item and
counter dtypes, per-thread item count, static bin count when one exists,
atomic-versus-sort algorithm selection, and the parameter roles for CUB's
`InitHistogram`, `Histogram`, and `Composite` member functions.
`make_block_histogram_spec` adds the static multidimensional block shape and
the exact CUB class-template arguments. A separate `INSTANCE` operation
records the parent object's specialization and temporary-storage identity
without pretending that constructing a parent placeholder invokes the
combined `Histogram(items, counters)` method. Its core signature contains only
temporary storage. Core validation rejects boolean or non-integral item and
bin counts consistently.

Numba-CUDA-MLIR keeps lazy placeholders, parent/child rewriting, linking, and
invocable construction while lowering core `INIT` and `COMPOSITE` specs.
CuTe requires static bins and exact launch dimensions. Group-first
`coop.histogram(group, ...)` and the private compatibility adapter build one
`GroupHistogramSemantics` plan and one public-CUB `BlockHistogram` artifact.
The wrapper owns CUB `TempStorage`, invokes `Histogram(items, counters)` once,
performs the required block barrier, and projects the shared counter array into
striped per-thread `ThreadData`. Runtime bins cannot instantiate CUB's `BINS`
template argument and therefore fail during tracing rather than selecting a
handwritten fallback.

Pure tests pin all lifecycle and member-operation signatures, algorithm
spellings, parameter roles, validation, and semantic identity. Adapter tests
pin Numba-CUDA-MLIR init/composite ABIs. CuTe tests pin root/private-adapter artifact
identity, exact multidimensional specialization, one CUB collective call, and
static-bin rejection; focused GPU coverage adds root/private-adapter numerical
identity for the one-shot path. The Numba-CUDA-MLIR one-shot and two-phase
paths retain their existing coverage.

## BlockRadixRank interval and prefix slice

The shared radix vocabulary separates ordering from bit-interval binding.
`RadixOrder` replaces frontend truthiness coercion with an explicit ascending
or descending choice. `RadixBitRange` records static-versus-runtime
classification without retaining traced values, validates every statically
known bound against the key width, and is intended for reuse by radix sort.
The import-free boolean normalizer recognizes native booleans and NumPy's
boolean scalar classes through their type hierarchy; this is deliberately a
small duck-typed compatibility shim in the shared block vocabulary rather than
a NumPy dependency in core.

`make_block_radix_rank_spec` owns CUB's eight class-template arguments, the
`RankKeys` member ABI, the `BFEDigitExtractor` construction, and the exact
exclusive-digit-prefix extent derived from radix width and block size. The
digit extractor keeps `KeyT` as a core dependency; Numba-CUDA-MLIR substitutes
its C++ spelling while adapting the descriptor.
This removes frontend-specific C++ type rendering and duplicate template
graphs. More generally, core `CxxFunction` dtype dependencies now substitute
their bracketed `<Name>` placeholder during adapter lowering, mirroring the
existing dependent-operator convention.

CUTLASS requires the rank interval to be trace-static, resolves the exact block
shape from launch facts, and builds a shared `AlgorithmSpec` through
`make_block_radix_rank_spec`. That spec materializes one official
`cub::BlockRadixRank` provider for root and private-adapter calls, including the exact
per-thread digit-prefix extent when that output is requested. The static bit
interval, order, dtype, item count, prefix shape, and block dimensions
participate in artifact identity; no runtime-width rank shim or handwritten
rank fallback remains.

For static widths of at least eight bits, the CUTLASS planner respecializes the
CUB shared-memory configuration to `cudaSharedMemBankSizeEightByte`. Packing
four 16-bit counters per word halves the memoized raking segment that otherwise
drives the eight-bit route to the register limit. Root and private-adapter CUTLASS calls
share that specialization and artifact; Numba-CUDA-MLIR retains the shared
core's four-byte configuration.

Numba-CUDA-MLIR rejects negative or out-of-width intervals before C++
generation, does not truncate fractional item counts, and does not coerce
arbitrary truthy values into descending order. Existing `BlockRadixRank`,
`RankKeys`, and `block_radix_rank` symbol components remain unchanged. Its
single-phase rewrite defaults to a four-bit window and rejects that default if
it would exceed the key width. Signed inputs retain their frontend ABI while a
generated local unsigned payload applies CUB's sign-bit transform before rank
extraction, matching CUTLASS and Radix Sort bit ordering.

The Numba-CUDA-MLIR two-phase rewrite consumes the shared interval-validation
and prefix-extent helpers while retaining backend-owned IR matching and
lowering. The backend-neutral spec can represent any interval accepted by CUB,
but the common, CUTLASS group-first, and Numba group-first APIs deliberately
cap the public rank width at eight bits. This shared boundary keeps prefix
extents and specialization costs bounded and makes the two certified backends
agree exactly. Private factory adapters retain their backend-specific
validation for compiler rewrites and regression coverage.

## BlockRadixSort payload, output, and bit-overload slice

`make_block_radix_sort_semantics` reuses the shared radix order and interval
records for keys-only and key/value sorts. It also records blocked versus
blocked-to-striped output and whether a wrapper exposes CUB's default-bit
overload, its explicit runtime-bit overload, or both. Known bounds are
validated before the record drops their payload values, so runtime arguments
do not fragment semantic or compiler-cache identity.
`make_block_radix_sort_spec` adds the block shape and owns all ten
`cub::BlockRadixSort` template arguments, the CUB defaults, member selection,
and overload ordering. The per-pass `RADIX_BITS` template argument remains the
fixed CUB default of four; unlike BlockRadixRank's bit interval, it is not a
publicly configurable frontend option. No existing frontend exposed a
per-pass radix-width override, so keeping the CUB default preserves the prior
API and specialization behavior. This template argument controls each radix
pass; `begin_bit` and `end_bit` independently select the runtime digit window.

Numba-CUDA-MLIR preserves its established two-overload invocable so
the same object accepts calls with or without runtime bounds. When MLIR factory
bounds are known, core validates them but still emits both call signatures.
CuTe's explicit-only form remains a distinct frontend contract. Order on the
Numba-CUDA-MLIR rewrite path is fixed by the selected ascending or descending
primitive node rather than a runtime user flag. Its typing layer retains
backend-owned checks over compiler types, so an invalid one-shot call may raise
`TypingError` before construction reaches the shared core validator. CUTLASS
resolves the exact block shape, builds the shared `AlgorithmSpec`, and
materializes one official `cub::BlockRadixSort` artifact for overlapping root
and private-adapter calls. Runtime `begin_bit` and `end_bit` values pass through that
provider ABI but remain outside semantic and compiler-cache identity. Tracing,
launch-width discovery, implementation-owned storage, provider registration,
and FFI calls remain backend-owned; there is no handwritten sort fallback.

Radix decomposers are absent from the public CUTLASS and Numba-CUDA-MLIR
surfaces because neither backend implements executable decomposer plumbing.
They can be added when a backend can lower and test the contract rather than
advertising an argument that always fails. Core validation rejects fractional
or boolean item counts, truthy non-boolean order/output flags, and statically
invalid bit intervals before code generation. Existing
`BlockRadixSort`, `Sort`, `SortDescending`, blocked-to-striped member names,
and `block_radix_sort` symbol components remain unchanged. These stricter
errors are intentional user-visible validation changes for callers that relied
on the older truncation or truthiness behavior.

## BlockRunLengthDecode lifecycle slice

`BlockRunLengthDecodeSemantics` separates the class specialization from the
stateful lifecycle that uses it. It owns the input-run and decoded-item shapes,
decoded-offset type, optional relative-offset output, default versus explicit
decoded-window selection, and whether the frontend exposes total decoded size.
`make_block_run_length_decode_spec` then selects one of three wrapper stages:
the public CUB constructor, the public `RunLengthDecode` member, or a fused
constructor-and-member driver. The constructor and member specs carry exactly
the seven public `cub::BlockRunLengthDecode` class-template arguments plus
auxiliary run-length, total-size, and relative-offset type bindings used to
specialize method parameters. The auxiliary bindings are deduced method types;
adapters do not append them to the CUB class specialization. Class rendering is
driven only by the `TemplateParameter` tuple; extra specialization-argument
keys substitute method parameters without widening the class template.
This slice models the length-initializing constructors used by all three
frontends. If a future frontend proposal exposes CUB's distinct run-offset
constructor family, it would need a separate core variant; this is design
guidance, not scheduled work or a binding roadmap commitment. Its absence of a
total-size output must not be represented by relaxing the length-constructor
contract.

The fused driver is not a substitute for a hidden or absent CUB API:
`cub::BlockRunLengthDecode` is public. Numba-CUDA-MLIR invocables represent one
call, while the CUB primitive requires construction followed by a member call
that reuses the same `TempStorage`. The core-owned driver preserves those two
public operations inside one MLIR invocable.

CUTLASS root and private compatibility forms build one
`GroupRunLengthDecodeSemantics` plan and one public-CUB artifact. The wrapper
owns `TempStorage`, invokes the core fused driver once, and synchronizes before
storage reuse. That driver constructs `cub::BlockRunLengthDecode` and calls its
public decode member with the same storage. Its explicit runtime window offset
is excluded from artifact identity. The wrapper performs only thread-local
postprocessing: targets beyond the decoded total become zero values and
length-typed all-ones relative-offset sentinels (the unsigned maximum or signed
`-1`). The CUTLASS ABI uses one
`length_type` for run lengths, decoded offsets, total size, and relative
offsets; independently typed offset outputs remain unsupported. Root and
private-adapter forms therefore share exact multidimensional block-shape identity and
the same generated CUB lifecycle rather than separate collective algorithms.
Actual run lengths must be positive, apart from one optional trailing
zero-padding suffix, and their positive block-wide sum must be representable in
the run-length dtype.

The shared contract stops coercing boolean or fractional run/item counts with
`int(...)` and rejects partial constructor operand sets before code generation.
Those inputs do not describe valid CUB specializations and receive an early
`ValueError`. The fused Numba-CUDA-MLIR rewrite and CuTe surface construct
complete operand sets by shape, so that invalid state is unrepresentable there.

## WarpReduce and WarpScan provider slice

`cuda.coop._core.warp.make_warp_reduce_spec` owns the scalar CUB WarpReduce
selection between custom `Reduce`, `Sum`, `Min`, and `Max`, together with the
logical-warp width, optional runtime valid-item count, operator descriptor,
temporary storage, and scalar-return ABI.

`cuda.coop._core.warp.make_warp_scan_spec` owns:

- exclusive versus inclusive mode;
- dedicated sum entry points versus explicit scan operators;
- full versus partial logical warps;
- optional static or runtime initial values;
- optional warp-aggregate output pointers; and
- scalar output returned through the backend-generated result slot.

Numba-CUDA-MLIR lowers these specs through its core adapter. CuTe first builds
the same scalar spec, then translates it into its
provider-specific `_WarpShimRequest`. Multi-item `ThreadData` composition stays
in CuTe: the provider combines each thread's items before the scalar CUB warp
call and expands the scan result afterward. This keeps the shared layer honest
about the CUB primitive while preserving CuTe's higher-level value container.

The pure tests cover method selection, logical-warp validation, operator and
initial-value classification, semantic identities, partial calls, aggregate
outputs, and adapter ABIs. Backend tests additionally compile and run the
Numba-CUDA-MLIR warp paths and render the CuTe provider bundles.

## WarpLoad and WarpStore data-movement slice

`cuda.coop._core.warp.make_warp_load_spec` and `make_warp_store_spec` own the
CUB algorithm mapping, logical-warp width, items per lane, full versus partial
tile signatures, optional load default value, pointer/array roles, and temporary
storage semantics. Frontends retain their public enum and alias normalization.

The Numba-CUDA-MLIR factory retains both the full-tile and partial-tile overloads
when partial support is requested. CuTe group-first calls and eligible
private-adapter calls with exact complete-physical-warp facts and a raw compact
pointer materialize the public-CUB plan. Private adapters collapse `vectorize`
to `direct` and reject `transpose` before route selection. Non-contiguous,
statically unproven, or CUB-incompatible private tensor routes retain the CuTe
indexing payload adapter; its tracing, predication, and element access remain
CuTe-owned.

## WarpExchange overload slice

`cuda.coop._core.warp.make_warp_exchange_spec` owns the CUB mapping for
striped-to-blocked, blocked-to-striped, and scatter-to-striped movement. It
records the logical-warp width, items per lane, shared-memory algorithm,
temporary storage, optional scatter rank dtype, and in-place versus
out-of-place overload shape.

Numba-CUDA-MLIR may retain both scatter overloads when the factory leaves
`use_output_items` unspecified; that option is rejected for non-scatter
modes. Complete physical-warp striped-to-blocked and blocked-to-striped calls
from the root or private adapter share one group plan and public
`cub::WarpExchange` provider. Logical-width and scatter forms remain private
compatibility routes and retain their four-item limit and `Int32` rank
contract. Provider registration, exact typed scratch, and generated C++ remain
in the CuTe backend.

The core uses one canonical `warp_exchange` C name and keeps the mode in the
method name. This intentionally replaces Numba-CUDA-MLIR's older mode-prefixed
C names; generated MLIR symbols and any compile artifacts keyed by the old
names must therefore be regenerated.

## WarpMergeSort payload and comparator slice

`cuda.coop._core.warp.make_warp_merge_sort_spec` owns the CUB
`WarpMergeSort::Sort` contract: keys-only versus key/value payloads, items per
lane, logical-warp width, full versus partial tile policy, in-place key/value
arrays, optional runtime `valid_items` and `oob_default`, temporary storage,
and the static comparison-operator descriptor. The backend-neutral `INT8`
dtype records the comparison callable's integral result without importing a
compiler type.

Numba-CUDA-MLIR materializes the shared spec directly and retains its callable
compilation, UDT wrapper, rewrite, and invocable paths.
CUTLASS materializes the same specialization through the reusable core adapter.
The static less/greater comparator, logical width, and tile policy participate
in artifact identity; runtime valid counts and out-of-bounds values do not.
Root calls accept only a complete physical warp, while private compatibility
adapters may select a complete logical width that exactly partitions the CTA.
Both routes support full and partial tiles. A partial valid count applies to
each warp tile, key/value association is preserved in its valid sorted prefix,
and outputs beyond that prefix are unspecified. One typed storage instance is
selected per logical warp before the single public `Sort` call.

Pure tests pin payload roles, template arguments, comparator identity, tile
policy, parameter roles, logical-warp validation, and semantic identity.
Backend tests pin the materialized Numba-CUDA-MLIR ABIs, CUTLASS source and
symbols, root/private-adapter artifact identity, full/partial physical and logical
routes, and GPU keys and pairs behavior.

Numba-CUDA-MLIR now de-duplicates link inputs for every cooperative primitive
by the full materialized algorithm identity, including the method, parameters,
and operator definitions. This is a global correction to the previous C-name
key: a C name can intentionally cover several methods, so the old key could
silently omit all but the first definition. Identical algorithm identities and
identical link-file paths still coalesce. Same-kernel WarpReduce, WarpScan, and
WarpExchange tests exercise multiple methods through this linker path.

## Thread hierarchy and group descriptors

`cuda.coop._core.ThreadHierarchy` owns the normalized grid, cluster, and block
shape model. `ThreadHierarchy.current()` represents an implicit hierarchy whose
runtime shape is backend-owned. `cuda.coop._core.ThreadGroup` records the group
level, hierarchy, static/current state, and stable semantic identity. The
backend fact source used to resolve static dimensions does not change group
equality or cache identity.

The CUTLASS frontend re-exports `ThreadHierarchy` directly and subclasses
`ThreadGroup` only to attach provider-backed `rank`, `count`, `is_member`,
`sync`, and `sync_aligned` operations. Hierarchy normalization and `this_*`
construction are not reimplemented by the backend. Core and backend-subclass group instances are
kept in their respective identity spaces; a backend does not mix the two group
classes in one cache namespace. Dataclass equality is type-aware, so otherwise
identical core and backend-subclass groups compare unequal.

An empty `coop.this_block()` or `coop.this_warp()` initially carries an implicit
hierarchy. Group-first collectives and hierarchy-dependent group methods require
static specialization shape, so the CuTe adapter resolves the group from
authoritative compiler launch facts, including the active kernel's exact NVVM
`reqntid` attribute, before creating its provider request. `maxntid` is only an
upper bound and is never used to specialize an operation. The generated block
shim therefore receives the exact one-, two-, or three-dimensional `cuda::block_dims`
specialization while user code keeps the natural current-group spelling. The
shared resolver also reconciles cluster/grid extents and verified launch
capabilities for all seven group forms.

The supported CuTe launch path emits `reqntid` from the kernel's launch block,
which is why the adapter can treat it as exact. If that attribute is absent,
inference fails closed with `NotImplementedError`; it never substitutes a
`maxntid` bound.

The full normalized block shape, not only its thread-count product, contributes
to the group semantic key and generated symbol. This matches the actual
`cuda::block_dims<x, y, z>` C++ type: launches with `(64, 1, 1)` and `(8, 4, 2)`
therefore produce distinct cached shims even though reduction values are
invariant for the same participating thread count. Recovered compiler launch
facts are authoritative; missing or invalid facts fail lowering instead of
being replaced by caller-specified dimensions.

## Adding another primitive

Before adding a builder:

1. Keep public argument aliases, normalization, and error text in each
   frontend.
2. Describe normalized semantics in core.
3. Distinguish semantic output from backend return behavior.
4. Exclude runtime operand values from specialization identity.
5. Lower through the adapter; do not import backend modules from core.
6. Preserve backend-owned compiler, provider, cache, rewrite, linker, and
   launcher paths.
7. Add pure core tests plus focused backend signature and compilation tests.

Choose the next migration for a missing semantic or ABI dimension.
