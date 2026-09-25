# `cuda.coop`

`cuda.coop` provides cooperative primitives for CUDA thread groups in Python
kernel DSLs. Its optional Numba-CUDA-MLIR and CUTLASS integrations support
Numba and CuTe kernels using CUB and CUDAX.

The distribution is a universal Python wheel containing a coherent bundle of
CUB, Thrust, libcu++, and CUDAX headers. Installed-wheel compilation uses that
bundle by default. Development from a CCCL source checkout uses the matching
checkout headers, and `CUDA_COOP_CCCL_ROOT` can select a different source
checkout or `cuda-coop` header bundle. None of these modes substitutes CUB
headers from the active CUDA Toolkit.

## Installation

Install `cuda-coop` without adding Python package dependencies:

```bash
python -m pip install cuda-coop
```

The wheel includes the common API, `cuda.coop.numba_mlir`,
`cuda.coop.cutlass`, type declarations, and bundled CCCL headers. The base
install declares no Python package dependencies. You can import `cuda.coop`
without a compiler or GPU; using an integration requires its backend
dependencies to be installed.

For Numba-CUDA-MLIR, choose the extra matching the CUDA Toolkit major version:

```bash
python -m pip install "cuda-coop[numba-cuda-mlir-cu13]"
# Use numba-cuda-mlir-cu12 with CUDA 12.
```

Both commands install the same `cuda-coop` wheel with the same DSL
integrations. The extra only adds the dependency requirements declared in
`pyproject.toml` so pip installs the supported Numba-CUDA-MLIR stack for the
selected CUDA major version.

Python 3.10 through 3.14 is supported. The Numba-CUDA-MLIR integration requires
`numba-cuda-mlir>=0.5.0,<0.6`.

Backend compiler and runtime CI is configured for Linux x86-64 with Python 3.14:
CUDA 13 in pull requests and CUDA 12 in the nightly matrix. The nightly
matrix also configures H100 runtime tests with serial synchronization race
checking under CUDA 13. Linux host contracts cover Python 3.10 and 3.14.
Windows checks build and import the universal wheel and verify its headers;
they do not execute the compiler
backend. Other combinations need separate runtime qualification. See the
[validation scope](https://nvidia.github.io/cccl/unstable/python/coop.html#coop-numba-validation)
for coverage and hardware requirements.

With Numba-CUDA-MLIR 0.5.0 through 0.5.3, keep a compiled kernel's dispatcher
and configured launch callables in their original CUDA context. Reuse on
another device or after context teardown is not qualified because cached
architecture or launch state can belong to the original context. The upstream
[context-isolation fix](https://github.com/NVIDIA/numba-cuda-mlir/pull/314)
must be released and qualified before relying on that reuse.

The CUTLASS integration is implemented with Linux and CUDA 13 as its initial
development target. A supported public CUTLASS package has not yet been
qualified, so there is no CUTLASS installation extra or supported minimum
version. A compatible CuTe compiler must provide the external LTO-IR linking
and compilation hooks described in its developer guide.

- Numba-CUDA-MLIR: [Programming Guide](https://nvidia.github.io/cccl/unstable/python/coop/programming_guide.html)
  and [Developer Guide](https://nvidia.github.io/cccl/unstable/python/coop/developer_overview.html).
- CUTLASS: [Programming Guide](https://nvidia.github.io/cccl/unstable/python/coop_cutlass.html)
  and [Developer Guide](https://nvidia.github.io/cccl/unstable/python/coop/cutlass_developer_guide.html).

## Backend registration and imports

Register the chosen backend on the host before compiling kernels. For
Numba-CUDA-MLIR:

```python
from cuda import coop

coop.register("numba-cuda-mlir")

from numba_cuda_mlir import cuda
```

For CUTLASS CuTe:

```python
from cuda import coop

coop.register("cutlass")

from cutlass import cute
```

`register` loads the backend and installs its compiler hooks regardless of
import order. Repeated calls are safe and return `None`. The Numba name also
accepts `"numba_cuda_mlir"`. Registration requires compatible dependencies;
it does not install packages. Installing an extra is separate from activating
a compiler in the running process.

Importing `cuda.coop` after either compiler runtime automatically registers a
compatible integration. Importing `cuda.coop` alone does not load either
compiler. Use explicit registration when a dependency or earlier notebook cell
already imported `cuda.coop`. Both integrations can be registered in one
process; common calls select the backend from the active compiler context.

The common API in `cuda.coop` describes operations independently of a compiler.
Its public entry points live in `cuda/coop/_core/api/`; the private `_core`
package also contains shared implementation. A common spelling does not
guarantee that every backend supports the operation.

For CUTLASS-only code, use the qualified namespace directly:

```python
import cuda.coop.cutlass as coop
```

This also registers the integration; a separate `register` call is unnecessary.
The host `cuda.coop.register` helper belongs to the common namespace.

For a module containing both DSLs, use distinct qualified aliases:

```python
import cuda.coop.numba_mlir as numba_coop
import cuda.coop.cutlass as cutlass_coop
```

Each import registers its backend. Call `numba_coop` from Numba kernels and
`cutlass_coop` from CuTe kernels. Examples comparing the APIs use `coop` for
common calls and the longer aliases for qualified calls; application code
need not import both namespaces for one backend. Aliasing dotted imports also
avoids rebinding `cuda`, which Numba examples use for `cuda.jit`.

Both qualified APIs retain shared signatures, string selectors, and inference
rules. Numba-CUDA-MLIR adds local-array payloads, memory namespaces, and device
callbacks. CUTLASS adds CuTe register conversions and qualified controls such
as warp Scan aggregates and scalar Shuffle. Custom operators and Scan prefix
callbacks are currently supported only by Numba-CUDA-MLIR.

Both integrations accept `ThreadData(..., alignment=None)`: use a compile-time
positive power of two in bytes to request minimum payload storage alignment,
or omit it to let the compiler choose. This does not assert alignment of Load
or Store memory operands. Results belong to the active compiler; a NumPy dtype
selector in a CuTe kernel still produces CuTe values.

The [FAQs](https://nvidia.github.io/cccl/unstable/python/coop/faqs.html) explain
namespace choices and temporary storage. The
[Glossary](https://nvidia.github.io/cccl/unstable/python/coop/glossary.html)
explains terms and concepts, including blocked and striped layouts.

## Primitive families

| Family | Entry points |
| --- | --- |
| Memory operations | `load`, `store` |
| Reduction | `reduce`, `sum`, `reduce_batched` |
| Scan | `scan`, `inclusive_scan`, `exclusive_scan`, `inclusive_sum`, `exclusive_sum` |
| Data rearrangement | `exchange`, `shuffle` |
| Comparison sorting | `merge_sort_keys`, `merge_sort_pairs` |
| Radix sorting and ranking | `radix_sort_keys`, `radix_sort_pairs`, `radix_rank` |
| Top-k selection | `topk_min_keys`, `topk_max_keys`, `topk_min_pairs`, `topk_max_pairs` |
| Neighbor comparisons | `adjacent_difference`, `discontinuity` |
| Counting | `histogram` |
| Run Length Decode | `run_length_decode`, `run_length_decode_into` |

Both backends implement Load/Store, Reduce/Sum, Scan, Exchange/Shuffle,
Merge Sort, Radix Sort/Rank, TopK, Adjacent Difference, and Discontinuity.
Histogram, Run Length Decode, and Batched Warp Reduction are currently
implemented only by Numba-CUDA-MLIR.

Each operation documents its supported groups and result ownership in the
[API reference](https://nvidia.github.io/cccl/unstable/python/coop_api.html).
The [visualizations](https://nvidia.github.io/cccl/unstable/python/coop/visualizations/index.html)
explain these contracts with interactive diagrams and tested kernel examples.
Histogram returns fresh counters that callers can accumulate in ordinary
payloads. Bulk Run Length Decode prepares and consumes its run table within
one call; neither API requires a persistent parent object.

## Configuration

Runtime configuration is controlled by these environment variables:

| Variable | Effect |
| --- | --- |
| `CUDA_COOP_DISABLE_AUTO_DSL_REGISTRATION` | A truthy value disables automatic backend activation during `cuda.coop` import. Explicit `coop.register(...)` and backend imports still work. |
| `CUDA_COOP_CCCL_ROOT` | Selects a CCCL source checkout or a `cuda-coop` header bundle. An invalid configured root is an error; resolution does not fall back to another CCCL source. |
| `CUDA_COOP_ENABLE_CACHE` | A truthy value enables the Numba-CUDA-MLIR persistent compiler cache. The value is read when its cache module is imported. |
| `XDG_CACHE_HOME` | For Numba-CUDA-MLIR on Linux and other POSIX systems, sets the cache base directory; entries are stored in `<value>/cccl`. Unset, empty, or relative values fall back to `~/.cache/cccl`. Read when the backend cache module is imported. |
| `LOCALAPPDATA` | For Numba-CUDA-MLIR on Windows, sets the cache base directory; entries are stored in `<value>\cccl`. Unset, empty, or relative values fall back to `~\AppData\Local\cccl`. Read when the backend cache module is imported. |
| `CUDA_COOP_CUTLASS_PROVIDER_CACHE_DIR` | Selects the CUTLASS provider artifact cache directory. The default is a user-specific directory under the system temporary directory; see the [CUTLASS Developer Guide](https://nvidia.github.io/cccl/unstable/python/coop/cutlass_developer_guide.html) for cache validation and artifact lifetime. |
| `CUDA_COOP_SOURCE_DUMP_DIR` | Writes generated CUDA source as `cuda_coop_<backend>_<hash>.cu` files. Set before compiling; provider cache hits in both integrations also dump source. Unset or empty disables dumping. |
| `CUDA_PATH` | Supplies `<value>/include` as a CUDA header candidate if `cuda-pathfinder` does not resolve one. |
| `CUDA_HOME` | Supplies `<value>/include` after `CUDA_PATH` under the same fallback rule. |
| `CUDA_ROOT` | Supplies `<value>/include` after `CUDA_HOME` under the same fallback rule. |

On POSIX, `/usr/local/cuda/include` is tried last. Windows uses
`cuda-pathfinder` or the configured toolkit roots above, without the Unix
fallback. Compilation reports a header-resolution error if none resolves
valid CUDA headers.

The build recognizes these CMake cache variables:

| Variable | Default | Effect |
| --- | --- | --- |
| `CUDA_COOP_INSTALL_HEADER_BUNDLE` | `ON` | Installs the private CCCL header and CMake-package bundle into the wheel. |
| `CUDA_COOP_ALLOW_DIRTY_HEADER_BUNDLE` | `OFF` | Allows a Git-worktree bundle when selected inputs are changed or `git status` cannot verify them, and records its source revision as `unknown`. |
| `CUDA_COOP_CCCL_SOURCE_REVISION` | empty | Supplies the revision token recorded instead of deriving it from Git. A dirty or unverifiable Git worktree still records `unknown`. |

For the two Boolean runtime switches, values are case-insensitive; `0`,
`false`, `no`, `off`, and the empty string are false.

## Block and Warp Load and Store

The common `cuda.coop` and qualified `numba_coop` and `cutlass_coop`
Load/Store calls share algorithm names, tile controls, and in-place Load
behavior. The following Numba kernel body clamps a grid tile tail, where
`source`, `destination`, and `count` are kernel arguments:

```python
from numba_cuda_mlir import cuda, types

from cuda import coop

block = coop.this_block()
items = coop.ThreadData(2)
tile_items = cuda.blockDim.x * 2
tile_offset = cuda.blockIdx.x * tile_items
valid_items = count - tile_offset
if valid_items < 0:
    valid_items = 0
elif valid_items > tile_items:
    valid_items = tile_items
coop.load(
    block,
    source,
    items,
    algorithm="direct",
    valid_items=valid_items,
    oob_default=0,
    offset=tile_offset,
)
coop.store(
    block,
    destination,
    items,
    algorithm="direct",
    valid_items=valid_items,
    offset=tile_offset,
)
```

The [CuTe tile-copy example](examples/cutlass/block_load_store.py) uses the
same controls with CuTe pointers. It includes allocation, launch, and a NumPy
reference check.

`load` fills the caller's output in place and returns `None`. `valid_items`
counts items across the selected group tile, while `offset` is a nonnegative
element offset. Runtime offsets are caller-validated. Numba source and
destination arrays must be one-dimensional and contiguous. CuTe operands may
be global-memory pointers or supported contiguous one-dimensional tensors. Without `oob_default`, invalid Load
slots retain their previous values. Every supplied runtime control
(`valid_items`, `oob_default`, and `offset`) must be uniform within its selected
group; different groups may use different values.

Runtime `valid_items` and `offset` accept signed integer types through 64 bits
and unsigned integer types through 32 bits. Boolean, floating-point, and
`uint64` runtime values are rejected. A runtime `oob_default` is already typed
by the compiler and must exactly match the Load payload dtype. Ordinary Python
integer and floating-point literals are converted contextually and checked
against that dtype before provider generation.

> **`valid_items` must satisfy
> `0 <= valid_items <= group_size * items_per_thread`.** Static values outside
> that range are rejected while planning. Runtime values are checked rather
> than saturated; do not rely on CUB's oversized-count behavior. An invalid
> runtime value executes a deterministic device trap before narrowing to CUB's
> integer parameter, and that trap poisons the current CUDA context. Clamp
> grid-stride and tail counts as above. Run intentional failure probes in
> disposable processes. For a Warp-group call, subtract that group's tile
> origin from a block-wide remainder and clamp the result to
> `[0, group_size * items_per_thread]`.

Store payloads must have exactly the destination dtype. Numba-CUDA-MLIR may
promote integer arithmetic even when its operands are 32-bit, so explicitly
cast computed values before storing them:

```python
value = types.int32(source[cuda.threadIdx.x] + 1)
coop.store(block, destination, value, algorithm="direct")
```

CuTe kernels have the same exact-dtype Store requirement; use a CUTLASS scalar
constructor such as `cutlass.Int32(expression)` when an explicit conversion
is needed. A NumPy dtype selector does not turn a traced CuTe scalar into a
host NumPy value.

Both common and qualified entry points use the same lowercase string
algorithm vocabulary: `direct`, `striped`, `vectorize`, `transpose`,
`warp_transpose`, and `warp_transpose_timesliced`. All six are executable.
`striped` exposes a striped per-thread payload; the other Load algorithms expose
blocked payloads. Store consumes the matching arrangement. The transpose Store
implementations copy their payload before calling CUB, so the caller's
`ThreadData` remains unchanged. The two warp-transpose modes require a block
size divisible by 32.

Algorithm selectors are normalized to lowercase underscore-delimited strings.
Enum and integer selectors, including `0`, are rejected.

Warp Load and Store accept `this_warp()` and support `direct`, `striped`,
`vectorize`, and `transpose`. Partition a physical warp into consecutive
logical groups with `this_warp().group_by(width)`, where `width` is 1, 2, 4, 8,
16, or 32. The enclosing block must contain a multiple of 32 threads and must
not have an incomplete final physical warp. Every member of a participating
group must reach the primitive; complete sibling logical groups may diverge.
`direct` and `vectorize` expose blocked payloads, `striped` exposes a striped
payload, and `transpose` uses striped memory transactions while exposing a
blocked payload.

Each Warp group addresses a distinct tile. The compiler advances the memory
base by `group_index * (group_size * items_per_thread)` and then applies the
caller's element `offset`. The offset must be uniform within each participating
group; different groups may use different offsets. The group index is the
x-major linear thread rank divided by the selected group size. For a
multi-block traversal, include the block's global tile origin in the caller
offset; the compiler-provided origin distinguishes the physical or logical
Warp groups within that block and must not be added again. Runtime offsets must
also leave enough signed 64-bit range for the last group origin in the block;
static offsets are checked during planning. `valid_items` is relative to each
group's own tile, not the entire block, and must be uniform within that group.

`ThreadGroup` follows the C++ hierarchy query surface. `rank(level="thread")`
and `count(level="thread")` accept `thread` (or `gpu_thread`), `warp`, `block`,
`cluster`, and `grid`. Their default result is the unsigned product type used
by the corresponding C++ hierarchy operation: normally `uint32`, and `uint64`
when the group or queried outer level is the grid. Use
`rank_as(dtype, level="thread")` or `count_as(dtype, level="thread")` to select
an explicit signed or unsigned 8-, 16-, 32-, or 64-bit integer dtype.
`is_member()` returns an integer membership flag.

These widths apply to both integrations. CuTe queries return `Uint32` or
`Uint64`, and `is_member()` returns `Uint8`; Numba queries return the matching
Numba integer types.

`sync()` and `sync_aligned()` expose the matching non-grid group barriers. All
participating members must reach `sync()`. `sync_aligned()` additionally
requires the caller to keep the group aligned and converged. Neither integration
supports grid synchronization. Numba-CUDA-MLIR cannot request a cooperative
grid launch; CUTLASS does not expose a grid synchronization implementation.

`group_by` remains static compiler vocabulary: `count` and `exhaustive` must be
compile-time constants. A logical threads-within-warp group can query its
threads and immediate parent Warp; a mapped warps-within-block group can query
its threads, physical Warps, and immediate parent block. Queries above the
immediate physical parent are rejected. Mapped warps-within-block groups expose
queries and `is_member()` but not `sync()` or `sync_aligned()`; their block
barrier lifetime must be owned by a future planner contract. For a
non-exhaustive partition, use `is_member()` to guard rank-dependent work for
excluded threads. Do not use that branch to skip a primitive unless the
primitive's participation contract explicitly permits it; every required
group or parent-group participant must still reach the primitive.

## Temporary storage

Block Load and Store accept an optional caller descriptor:

```python
storage = coop.TempStorage(
    size_in_bytes=None,
    alignment=None,
    auto_sync=None,
    sharing="shared",
)
coop.load(block, source, items, algorithm="transpose", temp_storage=storage)
```

For block, physical Warp, and logical Warp calls, `direct`, `striped`, and
`vectorize` are storage-free in both integrations: they need no shared-memory
allocation, storage pointer arguments, or reuse barriers. An explicit block
descriptor is validated but does not change that code generation.

The three block transpose algorithms use CUB temporary storage. Without a
descriptor, each compiler allocates the specialization's exact storage and
inserts a block reuse barrier. A descriptor selects shared or exclusive slices
and may request capacity and minimum alignment. The provider determines the
required byte count and alignment.

A descriptor's `sharing` selects only the slice layout: `"shared"` overlaps
calls that pass the same descriptor on one region; `"exclusive"` gives each
call site its own slice. A call site inside a loop reuses its slice under either
policy. `auto_sync` defaults to `True` for both policies and both integrations.

Distinct descriptors and compiler-owned storage do not alias each other. Each
scratch-using call appends a barrier after the operation, including the last
call. That barrier protects reuse of CUB scratch; it does not replace barriers
needed by the kernel's own shared-memory operations. With `auto_sync=False`,
call `storage.sync()` or the appropriate block barrier before reusing the
scratch, including on the next loop iteration. Compiler-owned scratch always
synchronizes.

Construct descriptors inside the kernel. Numba-CUDA-MLIR resolves descriptors
in its compiler passes; a descriptor may also be passed to a device helper
inlined into that kernel. CUTLASS records uses while tracing the CuTe kernel,
probes exact storage requirements, and allocates through CuTe's shared-memory
allocator before compilation finishes. The
[Numba storage example](tests/backends/numba_mlir/runtime/test_storage_examples.py)
and [CuTe storage example](examples/cutlass/block_storage.py) demonstrate
reuse policies and synchronization.

Numba-CUDA-MLIR has these additional compiler constraints:

- All cooperative storage in a kernel shares one backing. Above the 48 KiB
  static limit it moves to dynamic shared memory and the launch reserves the
  exact byte count. Supported Numba-CUDA-MLIR releases do not reliably separate
  static and dynamic allocations: a scratch-using kernel must not also declare
  a zero-sized or runtime-sized `cuda.shared.array`. When cooperative backing
  becomes dynamic, user static shared arrays are also unsupported. Storage-free
  operations do not add these restrictions.
- A descriptor with `auto_sync=False` must originate from one constructor site.
  Selecting between multiple manual-sync constructors is unsupported.
- Cooperative calls in device helpers must be inlined into the kernel. Use
  `@cuda.jit(device=True, inline="always")` when selecting the policy explicitly.
  Standalone primitive helpers and primitives inside standalone callbacks are
  unsupported. `literal_unroll` values cannot determine cooperative payload
  extents, group dimensions, selectors, or descriptor arguments. An unrelated
  `literal_unroll` loop does not add this restriction.

CuTe traces helper functions through its own compiler and allocates cooperative
scratch through its shared-memory allocator. For CuTe helper functions and
compiler-owned allocation, see the
[CUTLASS Programming Guide](https://nvidia.github.io/cccl/unstable/python/coop_cutlass.html)
and [Developer Guide](https://nvidia.github.io/cccl/unstable/python/coop/cutlass_developer_guide.html).

Warp `transpose` uses compiler-owned storage with one disjoint slice per
physical or logical group and masked synchronization for reuse. Both
integrations reject explicit `TempStorage` for every Warp Load and Store
algorithm, including the storage-free modes.

## Reduce and Sum

`sum(group, value, ...)` and `reduce(group, value, binary_op=..., ...)` return
one scalar with the payload element dtype. The common API accepts a numeric
scalar or fixed-size `ThreadData`; reducing a `ThreadData` payload combines all
items contributed by every participating member. The qualified
`numba_coop` API also accepts fixed-size `cuda.local.array` payloads;
`cutlass_coop` accepts CuTe register tensors and `TensorSSA` values.

A full built-in reduction has no `valid_items` or explicit `algorithm`. It uses
the storage-free CUDAX implementation for the current thread, a physical Warp,
a logical Warp from `this_warp().group_by(width)`, a block, a mapped group of
physical Warps from `this_block().group_by(warps_per_group)`, or a cluster.
Cluster reductions require matching cluster launch facts. Every member of the
selected group must participate.

By default, `broadcast=True` gives every group member the reduced scalar. With
`broadcast=False`, only rank zero of each selected group has a defined result;
other members must still execute the call and must not consume their returned
value. For example, this full block reduction combines two values per thread
but writes only from the block root:

```python
from numba_cuda_mlir import cuda, types

from cuda import coop


@cuda.jit
def block_sum(source, output):
    thread = cuda.threadIdx.x
    values = coop.ThreadData(2, dtype=types.int32)
    values[0] = source[2 * thread]
    values[1] = source[2 * thread + 1]
    total = coop.sum(coop.this_block(), values, broadcast=False)
    if thread == 0:
        output[0] = total
```

The complete Numba example is [block_sum.py](examples/numba_mlir/block_sum.py).
The [CuTe reduction example](examples/cutlass/reduce.py) also demonstrates
payload sums, logical-warp reductions, and root-only prefix results.

`sum` selects addition. `reduce` accepts the aliases `+`, `sum`, `add`, and
`plus`; `*`, `mul`, `multiply`, and `multiplies`; `min` and `minimum`; `max`
and `maximum`; and the bitwise pairs `&`/`bit_and`, `|`/`bit_or`, and
`^`/`bit_xor`. Bitwise reductions require an integer payload dtype. The
qualified APIs in both integrations additionally recognize the corresponding
Python `operator` functions and NumPy ufuncs. Built-in operator and algorithm selectors are
normalized to canonical lowercase strings. Enum-like and other non-string
selector objects are rejected.

Three controls select a direct CUB reduction instead:

- `valid_items` reduces the first N scalar values by linear group rank. It is
  available for block, physical-Warp, and logical-Warp groups and requires
  `broadcast=False`. N must be uniform within the group and satisfy
  `1 <= N <= group_size`. Static violations are rejected during compilation;
  runtime violations execute a deterministic device trap before CUB's 32-bit
  parameter is formed, invalidating the current CUDA context.
- `algorithm` selects block-only `raking_commutative_only`, `raking`, or
  `warp_reductions`, also with `broadcast=False`. Scalar and fixed-array
  payloads are supported. `raking_commutative_only` is restricted to Sum and
  recognized commutative built-ins. The addition-specific nondeterministic CUB
  variant is intentionally not exposed.
- A custom Python device callback is available only through
  `cuda.coop.numba_mlir.reduce`, must be stateless, and requires
  `broadcast=False`. It uses CUB for block, physical-Warp, or logical-Warp
  groups. Warp callbacks accept scalar payloads; block callbacks may also
  reduce fixed arrays. CUTLASS supports built-in reductions only. Stateful
  reduction callbacks are unsupported in both integrations.

Full CUDAX reductions have no external temporary-storage ABI, backing
allocation, or compiler-inserted post-call barrier; the primitive call still
requires converged group participation. Direct CUB reductions use
compiler-owned shared storage. Block paths append a block reuse barrier, while
physical and logical Warp paths append `syncwarp` for the exact participating
mask. Reduce and Sum do not currently accept caller `TempStorage` descriptors.

Grid Reduce and Sum are unsupported because a grid reduction requires hidden
per-launch workspace; use a separate kernel or explicitly managed multi-stage
reduction instead.

## Scan

The Scan family has five spellings: `scan`, `exclusive_scan`,
`inclusive_scan`, `exclusive_sum`, and `inclusive_sum`. `scan` selects its form
with `mode="exclusive"` or `mode="inclusive"`; the other names make that choice
explicit. Every form returns a fresh scalar or per-thread payload and leaves
the input unchanged.

Block Scan accepts a numeric scalar or fixed-size `ThreadData`. Qualified
`numba_coop` calls also accept fixed-size `cuda.local.array` payloads;
`cutlass_coop` calls accept CuTe register tensors and `TensorSSA` values. The
`raking`, `raking_memoize`, and `warp_scans` algorithms are available for
blocks. Physical- and logical-Warp Scan accept one scalar per thread and have
no algorithm selector.

Sum is the default operation. `scan`, `exclusive_scan`, and `inclusive_scan`
accept the same built-in string aliases as Reduce. Both qualified APIs also
recognize the corresponding Python `operator` functions and NumPy ufuncs.
Numba-CUDA-MLIR additionally accepts stateless device callbacks; CUTLASS does
not support custom scan operators. A non-sum exclusive scan requires an
`initial_value` matching the payload dtype; ordinary Python literals are
checked and converted in that context. A Numba block-prefix callback can
supply that prefix instead. Inclusive scans reject an initial value. The aggregate reports
only the input reduction and does not include the exclusive initial value.

Both qualified APIs accept `aggregate_output`, a one-item `ThreadData`
populated with the group aggregate. Its dtype must match the input or be
omitted for inference. Numba-CUDA-MLIR also accepts a one-item local array;
CUTLASS requires writable `ThreadData` for this output. Warp forms also
accept `valid_items`, which selects the first N lanes by group rank and requires
`1 <= N <= warp_width`; only those N result lanes are defined. The initial
value and `valid_items` must be uniform across participating members. An
out-of-range runtime `valid_items` value triggers a deterministic device trap
before CUB's 32-bit parameter is formed, invalidating the current CUDA context.

All five Numba-qualified Block Scan spellings accept a block-prefix callback
through the `prefix_op` keyword. CUTLASS does not support prefix callbacks or
callback state. A stateless callback receives the block aggregate and
returns the prefix:

```python
from numba_cuda_mlir import cuda, types

import cuda.coop.numba_mlir as numba_coop


@cuda.jit(device=True)
def prefix_after_aggregate(block_aggregate):
    return block_aggregate + 7


# Inside a kernel:
scanned = numba_coop.exclusive_sum(
    numba_coop.this_block(),
    value,
    prefix_op=prefix_after_aggregate,
)
```

For a running prefix, wrap a two-argument device callback in
`StatefulFunction`. Its first argument is a one-item state payload and its
second is the block aggregate. Pass the state as the third positional
argument:

```python
@cuda.jit(device=True)
def carry_prefix(state, block_aggregate):
    previous = state[0]
    state[0] = previous + block_aggregate
    return previous


running_prefix = numba_coop.StatefulFunction(carry_prefix, types.int64)

# Inside a kernel, before a loop over tiles:
state = numba_coop.ThreadData(1, dtype=types.int64)
state[0] = types.int64(0)
scanned = numba_coop.exclusive_sum(
    numba_coop.this_block(),
    value,
    state,
    prefix_op=running_prefix,
)
```

The state may be a numeric one-item `ThreadData` or local array. Its dtype must
exactly match the `StatefulFunction` descriptor, but may differ from the scan
payload dtype. Keep the same state object alive across repeated calls and give
every participating thread the same initial contents. CUB may invoke the
callback in every lane of the block's first warp, but only lane 0's returned
prefix is applied; only thread 0's state is authoritative after the calls.

Prefix callbacks are available only through Numba-qualified Block Scan. They are
mutually exclusive with `initial_value` and `aggregate_output`, are not
stateful binary `scan_op` values, and do not support Warp Scan, `valid_items`,
or structured state.

All Scan providers use CUB temporary storage. Block calls may use implicit or
caller-owned `TempStorage` and append a block reuse barrier unless
a caller-owned descriptor explicitly sets `auto_sync=False`. Physical and
logical Warp calls use compiler-owned per-Warp storage and append `syncwarp`
for the exact participating mask. Prefix callbacks retain the same storage
rules. When repeated calls reuse Block Scan storage, keep automatic
synchronization enabled or call `storage.sync()` before reuse when
`auto_sync=False`. The prefix state is persistent per-thread data, not CUB
temporary storage.

This example uses the common API to load a block tile, compute its exclusive
sum, and store the out-of-place result:

```python
import numpy as np
from numba_cuda_mlir import cuda

from cuda import coop


@cuda.jit
def block_scan_kernel(values, prefixes):
    block = coop.this_block()
    items = coop.ThreadData(2, dtype=np.int32)
    coop.load(block, values, items)
    scanned = coop.exclusive_sum(block, items)
    coop.store(block, prefixes, scanned)
```

The complete Numba example is [block_scan.py](examples/numba_mlir/block_scan.py).
The [CuTe Scan example](examples/cutlass/scan.py) performs the same tile
composition with an explicit initial value and shared scratch.

## Exchange and Shuffle

`exchange(group, value, mode=...)` returns a fresh payload and leaves `value`
unchanged. The common API accepts `striped_to_blocked` and
`blocked_to_striped` for block, physical Warp, and logical Warp groups. A
blocked tile gives each thread consecutive items. A striped tile gives item
`i` to lane `i % group_size` at per-thread position `i // group_size`.

Both `numba_coop.exchange` and `cutlass_coop.exchange` expose the
block-only `warp_striped_to_blocked` and `blocked_to_warp_striped` layouts and
the CUB scatter modes. Scatter ranks are local to the selected group tile and
must use a signed integer payload with the same extent as `value`.
Both accept `ThreadData`; qualified Numba calls also accept local arrays, and
qualified CUTLASS calls accept CuTe register payloads. Unguarded ranks must be in
`[0, group_size * items_per_thread)`. Guarded scatter skips negative ranks;
every nonnegative rank must still be in range. Flagged scatter uses only ranks
whose corresponding non-boolean integer flag is nonzero; each active rank must
be in range. Active destinations must be unique for a deterministic result;
holes and duplicate destinations are otherwise unspecified.
`warp_time_slicing=True` is available only for block Exchange and is not valid
for guarded or flagged scatter.

`shuffle(block, value, mode=...)` is block-only. The common API accepts a
`ThreadData` payload, `up` or `down`, and the fixed distance `1`; the vacated
edge item is unspecified. Both qualified APIs also accept scalar `offset`
and `rotate` modes. Offset distance is signed, may vary by thread, and must fit a
signed 32-bit integer. Static overflows are rejected during compilation;
runtime overflows trap before narrowing to CUB. Within that range, a source
rank outside the block leaves that thread's result unspecified. Rotate
distance may be static or runtime and must satisfy
`0 < distance < block_threads`. An invalid runtime Rotate distance also
executes a device trap. A trap invalidates that CUDA context, so validate
untrusted distances before launch.

Exchange and Shuffle require converged participation by every member of the
selected group. They use compiler-owned CUB temporary storage and append a
reuse barrier after every call. Block operations use one block-wide storage
instance and `syncthreads`; physical and logical Warp Exchange use one
disjoint slice per group and `syncwarp` with the exact group mask.

The [Numba rearrangement examples](tests/backends/numba_mlir/runtime/test_rearrangement_examples.py)
and [CuTe Exchange/Shuffle example](examples/cutlass/exchange_shuffle.py)
demonstrate these layouts and payload-preserving operations.

These APIs are compile-time kernel constructs. Calling them outside a
compatible compiler context reports a structured context error.

See the [CCCL documentation](https://nvidia.github.io/cccl/unstable/python/coop.html)
for the complete signatures.
