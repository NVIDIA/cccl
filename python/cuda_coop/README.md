# `cuda.coop`

`cuda.coop` provides portable cooperative Load and Store constructs for CUDA
thread blocks, physical warps, and logical warps in Python kernel DSLs. The
first backend targets Numba-CUDA-MLIR and lowers the operations to CUB.

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
slots retain their previous values.

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
caller's element `offset`. The group index is the x-major linear thread rank
divided by the selected group size. For a multi-block traversal, include the
block's global tile origin in the caller offset; the compiler-provided origin
distinguishes the physical or logical Warp groups within that block and must
not be added again. Runtime offsets must also leave enough signed 64-bit range
for the last group origin in the block; static offsets are checked during
planning. `valid_items` is relative to each group's own tile, not the entire
block, and must be uniform within that group.

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

These APIs are compile-time kernel constructs. Calling them outside a
compatible compiler context reports a structured context error.

See the [CCCL documentation](https://nvidia.github.io/cccl/unstable/python/coop.html)
for the complete signatures.
