# `cuda.coop`

`cuda.coop` provides portable cooperative data-movement and reduction
constructs for CUDA thread groups in Python kernel DSLs. The first backend
targets Numba-CUDA-MLIR and lowers data movement to CUB and hierarchy-aware
reductions to CUDAX or CUB.

The distribution is a universal Python wheel containing a coherent bundle of
CUB, Thrust, libcu++, and CUDAX headers. Installed-wheel compilation uses that
bundle by default. Development from a CCCL source checkout uses the matching
checkout headers, and `CUDA_COOP_CCCL_ROOT` can select a different source
checkout or `cuda-coop` header bundle. None of these modes substitutes CUB
headers from the active CUDA Toolkit.

## Installation

Choose the extra matching the CUDA Toolkit major version:

```bash
python -m pip install "cuda-coop[numba-cuda-mlir-cu12]"
python -m pip install "cuda-coop[numba-cuda-mlir-cu13]"
```

Python 3.10 through 3.14 is supported. Importing the portable `cuda.coop`
package does not require loading a compiler backend.

The Numba backend is intentionally limited to
`numba-cuda-mlir>=0.5.0,<0.6`. It currently uses a guarded compatibility shim
for private 0.5.x compiler registration APIs, so another runtime series is
rejected before compiler registries are changed. Replacing that shim with an
upstream public API is follow-up work.

When using the portable namespace with Numba-CUDA-MLIR, import the compiler
runtime first so `cuda.coop` can activate its compiler hooks automatically:

```python
from numba_cuda_mlir import cuda

from cuda import coop
```

A standalone `cuda.coop` import does not discover or load optional compiler
runtimes or CUDA bindings. If `cuda.coop` was imported first, explicitly import
the qualified backend before compiling a kernel:

```python
from cuda import coop
import cuda.coop.numba_mlir as _coop_numba_mlir  # Activate portable calls.
```

Importing `cuda.coop` first and compiling without that explicit activation is
unsupported. Numba-CUDA-MLIR then reports the portable marker as unknown
(typically `Unknown attribute 'this_block'`) because its compiler hooks were
not registered.

Keep the alias on the qualified activation import. A bare
`import cuda.coop.numba_mlir` binds the name `cuda` in the importing scope. If
that name already refers to the object imported by
`from numba_cuda_mlir import cuda`, the bare import replaces it and later
`@cuda.jit` uses the wrong module.

Using `import cuda.coop.numba_mlir as coop` instead activates the backend and
selects its qualified namespace. Shared operations retain the portable
signatures, string selectors, and inference rules; the qualified namespace
adds backend memory namespaces and the `ThreadData(..., alignas=...)`
payload-alignment control.

## Configuration

Runtime configuration is controlled by these environment variables:

| Variable | Effect |
| --- | --- |
| `CUDA_COOP_DISABLE_AUTO_DSL_REGISTRATION` | A truthy value disables automatic backend activation during `cuda.coop` import. Explicit qualified-backend import still works. |
| `CUDA_COOP_CCCL_ROOT` | Selects a CCCL source checkout or a `cuda-coop` header bundle. An invalid configured root is an error; resolution does not fall back to another CCCL source. |
| `CUDA_COOP_ENABLE_CACHE` | A truthy value enables the persistent compiler cache under `~/.cache/cccl`. The value is read when the backend cache module is imported. |
| `CUDA_COOP_NUMBA_MLIR_NVRTC_DUMP_DIR` | Writes content-addressed pre-NVRTC CUDA source files to this directory for compiler diagnostics. |
| `CUDA_PATH` | Supplies `<value>/include` as a CUDA header candidate if `cuda-pathfinder` does not resolve one. |
| `CUDA_HOME` | Supplies `<value>/include` after `CUDA_PATH` under the same fallback rule. |
| `CUDA_ROOT` | Supplies `<value>/include` after `CUDA_HOME` under the same fallback rule. |

If those mechanisms do not resolve CUDA headers, `/usr/local/cuda/include` is
tried last.

The build recognizes these CMake cache variables:

| Variable | Default | Effect |
| --- | --- | --- |
| `CUDA_COOP_INSTALL_HEADER_BUNDLE` | `ON` | Installs the private CCCL header and CMake-package bundle into the wheel. |
| `CUDA_COOP_ALLOW_DIRTY_HEADER_BUNDLE` | `OFF` | Allows a Git-worktree bundle when selected inputs are changed or `git status` cannot verify them, and records its source revision as `unknown`. |
| `CUDA_COOP_CCCL_SOURCE_REVISION` | empty | Supplies the revision token recorded instead of deriving it from Git. A dirty or unverifiable Git worktree still records `unknown`. |

For the two Boolean runtime switches, values are case-insensitive; `0`,
`false`, `no`, `off`, and the empty string are false.

## Block and Warp Load and Store

The portable `cuda.coop` entry points and the qualified
`cuda.coop.numba_mlir` entry points have matching signatures. The following
kernel-body example clamps a grid tile tail, where `source`, `destination`, and
`count` are kernel arguments:

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
loaded = coop.load(
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
    loaded,
    algorithm="direct",
    valid_items=valid_items,
    offset=tile_offset,
)
```

`load` returns the same output object passed by the caller. `valid_items` counts
items across the selected group tile, while `offset` is a nonnegative element
offset. Runtime offsets are caller-validated. Source and destination arrays
must be one-dimensional and contiguous. Without `oob_default`, invalid Load
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

Both portable and qualified entry points use the same lowercase string
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
group must reach the collective; complete sibling logical groups may diverge.
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

`ThreadGroup` objects are descriptor-only in this release. Runtime query,
membership, and synchronization methods such as `rank`, `count`, `rank_as`,
`count_as`, `sync`, `sync_aligned`, and `is_member` are not exposed.

## Temporary storage

Block Load and Store accept an optional caller descriptor:

```python
storage = coop.TempStorage(
    size_in_bytes=None,
    alignment=None,
    auto_sync=None,
    sharing="shared",
)
coop.load(block, source, items, temp_storage=storage)
```

For block, physical Warp, and logical Warp operations, `direct`, `striped`, and
`vectorize` are storage-free: they default-construct the CUB primitive and emit
no shared-memory allocation, pointer argument, or barrier. For a block call, an
explicit descriptor, including an unsized descriptor, is validated as
compile-time vocabulary but does not change code generation for those
algorithms.
Construct `TempStorage` inside the kernel; the current Numba-CUDA-MLIR frontend
does not resolve module-global storage descriptors.

The three block transpose algorithms use CUB temporary storage. Without a
descriptor, the compiler allocates the specialization's exact storage and
inserts a block reuse barrier. A caller descriptor can instead select shared or
exclusive ownership, request a capacity and alignment, or opt into dynamic
shared memory. The generated provider remains the authority for the required
byte count and alignment.

Warp `transpose` uses compiler-owned storage with one disjoint slice per
physical or logical group and inserts `syncwarp` with the exact group mask.
Explicit `TempStorage` is rejected by both the portable and qualified APIs for
every Warp Load and Store algorithm, including the storage-free modes.

## Reduce and Sum

`sum(group, value, ...)` and `reduce(group, value, binary_op=..., ...)` return
one scalar with the payload element dtype. The portable API accepts a numeric
scalar or fixed-size `ThreadData`; reducing a `ThreadData` payload combines all
items contributed by every participating member. The qualified
`cuda.coop.numba_mlir` API also accepts fixed-size `cuda.local.array` payloads.

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

The complete runnable form is in `examples/numba_mlir/block_sum.py`.

`sum` selects addition. `reduce` accepts the aliases `+`, `sum`, `add`, and
`plus`; `*`, `mul`, `multiply`, and `multiplies`; `min` and `minimum`; `max`
and `maximum`; and the bitwise pairs `&`/`bit_and`, `|`/`bit_or`, and
`^`/`bit_xor`. Bitwise reductions require an integer payload dtype. The
qualified API additionally recognizes the corresponding Python `operator`
functions and NumPy ufuncs. Built-in operator and algorithm selectors are
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
  reduce fixed arrays. Stateful callbacks and their per-launch state plumbing
  are deferred.

Full CUDAX reductions have no external temporary-storage ABI, backing
allocation, or compiler-inserted post-call barrier; the collective call still
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

Block Scan accepts a numeric scalar, fixed-size `ThreadData`, or, in the
qualified `cuda.coop.numba_mlir` API, a fixed-size `cuda.local.array`. The
`raking`, `raking_memoize`, and `warp_scans` algorithms are available for
blocks. Physical- and logical-Warp Scan accept one scalar per thread and have
no algorithm selector.

Sum is the default operation. `scan`, `exclusive_scan`, and `inclusive_scan`
accept the same built-in string aliases as Reduce. The qualified API also
recognizes the corresponding Python `operator` functions and NumPy ufuncs, and
accepts stateless device callbacks. A non-sum exclusive scan requires an
`initial_value` matching the payload dtype; ordinary Python literals are
checked and converted in that context. A block-prefix callback can supply that
prefix instead. Inclusive scans reject an initial value. The aggregate reports
only the input reduction and does not include the exclusive initial value.

The portable root API intentionally exposes only the common surface above. The
qualified API additionally accepts `aggregate_output`, an exact-dtype one-item
`ThreadData` or local array populated with the group aggregate. Warp forms also
accept `valid_items`, which selects the first N lanes by group rank and requires
`1 <= N <= warp_width`; only those N result lanes are defined. The initial
value and `valid_items` must be uniform across participating members. Invalid
runtime values execute a deterministic device trap before CUB's 32-bit
parameter is formed, invalidating the current CUDA context.

All five qualified Block Scan spellings accept a block-prefix callback through
the `prefix_op` keyword. A stateless callback receives the block aggregate and
returns the prefix:

```python
from numba_cuda_mlir import cuda, types

import cuda.coop.numba_mlir as coop


@cuda.jit(device=True)
def prefix_after_aggregate(block_aggregate):
    return block_aggregate + 7


# Inside a kernel:
scanned = coop.exclusive_sum(
    coop.this_block(),
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


running_prefix = coop.StatefulFunction(carry_prefix, types.int64)

# Inside a kernel, before a loop over tiles:
state = coop.ThreadData(1, dtype=types.int64)
state[0] = types.int64(0)
scanned = coop.exclusive_sum(
    coop.this_block(),
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

Prefix callbacks are available only through qualified Block Scan. They are
mutually exclusive with `initial_value` and `aggregate_output`, are not
stateful binary `scan_op` values, and do not support Warp Scan, `valid_items`,
or structured state.

All Scan providers use CUB temporary storage. Block calls may use implicit,
caller-owned, or dynamic `TempStorage` and append a block reuse barrier unless
a caller-owned descriptor explicitly sets `auto_sync=False`. Physical and
logical Warp calls use compiler-owned per-Warp storage and append `syncwarp`
for the exact participating mask. Prefix callbacks retain the same storage
rules. When repeated calls reuse Block Scan storage, keep automatic
synchronization enabled or issue `cuda.syncthreads()` after each call when
`auto_sync=False`. The prefix state is persistent per-thread data, not CUB
temporary storage.

This portable example loads a block tile, computes its exclusive sum, and
stores the out-of-place result:

```python
import numpy as np
from numba_cuda_mlir import cuda

from cuda import coop


@cuda.jit
def block_scan_kernel(values, prefixes):
    block = coop.this_block()
    items = coop.ThreadData(2, dtype=np.int32)
    loaded = coop.load(block, values, items)
    scanned = coop.exclusive_sum(block, loaded)
    coop.store(block, prefixes, scanned)
```

The complete runnable form is in `examples/numba_mlir/block_scan.py`.

## Exchange and Shuffle

`exchange(group, value, mode=...)` returns a fresh payload and leaves `value`
unchanged. The portable API accepts `striped_to_blocked` and
`blocked_to_striped` for block, physical Warp, and logical Warp groups. A
blocked tile gives each thread consecutive items. A striped tile gives item
`i` to lane `i % group_size` at per-thread position `i // group_size`.

The qualified `cuda.coop.numba_mlir.exchange` API additionally exposes the
block-only `warp_striped_to_blocked` and `blocked_to_warp_striped` layouts and
the CUB scatter modes. Scatter ranks are local to the selected group tile and
must use a signed integer `ThreadData` or local-array payload with the same
extent as `value`. Unguarded ranks must be in
`[0, group_size * items_per_thread)`. Guarded scatter skips negative ranks;
every nonnegative rank must still be in range. Flagged scatter uses only ranks
whose corresponding non-boolean integer flag is nonzero; each active rank must
be in range. Active destinations must be unique for a deterministic result;
holes and duplicate destinations are otherwise unspecified.
`warp_time_slicing=True` is available only for block Exchange and is not valid
for guarded or flagged scatter.

`shuffle(block, value, mode=...)` is block-only. The portable API accepts a
`ThreadData` payload, `up` or `down`, and the fixed distance `1`; the vacated
edge item is unspecified. The qualified API also accepts scalar `offset` and
`rotate` modes. Offset distance is signed, may vary by thread, and must fit a
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

These APIs are compile-time kernel constructs. Calling them outside a
compatible compiler context reports a structured context error.

See the [CCCL documentation](https://nvidia.github.io/cccl/unstable/python/coop.html)
for the complete signatures.
