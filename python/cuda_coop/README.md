# `cuda.coop`

`cuda.coop` provides cooperative primitives for CUDA thread groups in Python
kernel DSLs. The first backend targets Numba-CUDA-MLIR. The table below
lists the available operations.

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

The wheel includes the common API, every shipped DSL integration (including
`cuda.coop.numba_mlir`), type declarations, and bundled CCCL headers. The base
install declares no Python package dependencies. You can import `cuda.coop`
without a compiler or GPU; using an integration requires its backend
dependencies to be installed. Numba-CUDA-MLIR is the first supported backend;
CUTLASS support is planned.

For Numba-CUDA-MLIR, choose the extra matching the CUDA Toolkit major version:

```bash
python -m pip install "cuda-coop[numba-cuda-mlir-cu13]"
# Use numba-cuda-mlir-cu12 with CUDA 12.
```

Both commands install the same `cuda-coop` wheel with the same DSL
integrations. The extra only adds the dependency requirements declared in
`pyproject.toml` so pip installs the supported Numba-CUDA-MLIR stack for the
selected CUDA major version.

Python 3.10 through 3.14 is supported. The current backend integration requires
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

## Backend registration and imports

Register the backend on the host before compiling kernels:

```python
from cuda import coop

coop.register("numba-cuda-mlir")

from numba_cuda_mlir import cuda
```

`register` loads the backend and installs its compiler hooks regardless of
import order. It also accepts `"numba_cuda_mlir"`. Repeated calls are safe and
return `None`. It requires installed backend dependencies and does not install
packages. Installing an extra is separate from registering a compiler in the
running process; installed package metadata does not reliably record which
extra was requested.

Importing `cuda.coop` after `numba_cuda_mlir` also registers the backend
automatically. A standalone `cuda.coop` import does not load optional
compilers. Explicit registration works when a dependency or earlier notebook
cell already imported `cuda.coop`.

The common API exposed by `cuda.coop` describes operations independently of
a particular compiler backend. Its public entry points are implemented in
`cuda/coop/_core/api/`; the private `_core` package also contains shared
implementation used by the backends. Each backend implements its supported
operations, so a common spelling does not guarantee support in every compiler.

To use Numba-specific features, import the qualified backend namespace; this
also registers it:

```python
import cuda.coop.numba_mlir as numba_coop
```

Use `numba_coop` alongside common calls. In a program using only this backend,
`import cuda.coop.numba_mlir as coop` is also supported. Keep the alias: a bare
`import cuda.coop.numba_mlir` binds `cuda` to the top-level Python package and
can replace the name used for Numba's `cuda.jit`.

Shared operations retain the common signatures, string selectors, and
inference rules. The backend namespace adds Numba local-array payloads and
memory namespaces.
Both namespaces accept `ThreadData(..., alignment=None)`: use a compile-time
positive power of two in bytes to request minimum payload storage alignment,
or omit it to let the compiler choose. This does not assert alignment of Load
or Store arrays.

The [FAQs](https://nvidia.github.io/cccl/unstable/python/coop/faqs.html) explain
namespace choices and temporary storage. The
[Glossary](https://nvidia.github.io/cccl/unstable/python/coop/glossary.html)
explains terms and concepts, including blocked and striped layouts.

## Primitive families

| Family | Entry points |
| --- | --- |
| Memory operations | `load`, `store` |

Each operation documents its supported groups and result ownership in the
[API reference](https://nvidia.github.io/cccl/unstable/python/coop_api.html).
The [visualizations](https://nvidia.github.io/cccl/unstable/python/coop/visualizations/index.html)
explain these contracts with interactive diagrams and tested kernel examples.


## Configuration

Runtime configuration is controlled by these environment variables:

| Variable | Effect |
| --- | --- |
| `CUDA_COOP_DISABLE_AUTO_DSL_REGISTRATION` | A truthy value disables automatic backend activation during `cuda.coop` import. Explicit qualified-backend import still works. |
| `CUDA_COOP_CCCL_ROOT` | Selects a CCCL source checkout or a `cuda-coop` header bundle. An invalid configured root is an error; resolution does not fall back to another CCCL source. |
| `CUDA_COOP_ENABLE_CACHE` | A truthy value enables the persistent compiler cache. The value is read when the backend cache module is imported. |
| `XDG_CACHE_HOME` | On Linux and other POSIX systems, sets the cache base directory; entries are stored in `<value>/cccl`. Unset, empty, or relative values fall back to `~/.cache/cccl`. Read when the backend cache module is imported. |
| `LOCALAPPDATA` | On Windows, sets the cache base directory; entries are stored in `<value>\cccl`. Unset, empty, or relative values fall back to `~\AppData\Local\cccl`. Read when the backend cache module is imported. |
| `CUDA_COOP_SOURCE_DUMP_DIR` | Writes generated CUDA source as `cuda_coop_<backend>_<hash>.cu` files. Set before compiling; Numba provider cache hits also dump source. Unset or empty disables dumping. |
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

The common `cuda.coop` entry points and the qualified
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

`load` fills the caller's output in place and returns `None`. `valid_items`
counts items across the selected group tile, while `offset` is a nonnegative
element offset. Runtime offsets are caller-validated. Source and destination arrays
must be one-dimensional and contiguous. Without `oob_default`, invalid Load
slots have unspecified values, even if initialized before Load. Every supplied runtime control
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

Both common and qualified entry points use the same lowercase string
algorithm vocabulary: `direct`, `striped`, `vectorize`, `transpose`,
`warp_transpose`, and `warp_transpose_timesliced`. All six are executable.
`striped` exposes a striped per-thread payload; the other Load algorithms expose
blocked payloads. Store consumes the matching arrangement. As in CUB,
transpose Store algorithms may rearrange the input payload in place. Copy
values before Store if they are needed later. The two warp-transpose modes
require a block size divisible by 32.

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
    auto_sync=False,
    sharing="shared",
)
coop.load(block, source, items, algorithm="transpose", temp_storage=storage)
```

For block, physical Warp, and logical Warp calls, `direct`, `striped`, and
`vectorize` are storage-free: they default-construct CUB primitives without shared-memory allocation, pointer arguments, or
barriers. For block calls, an explicit descriptor is validated but does not
change their code generation. Construct `TempStorage` inside the kernel; module-global storage
descriptors cannot be resolved. A descriptor may be passed to a device helper
that Numba-CUDA-MLIR inlines into the kernel, which is the default.

The three block transpose algorithms use CUB temporary storage. Without a descriptor,
the compiler allocates the specialization's exact storage and inserts a block
reuse barrier. A caller descriptor can select shared or exclusive ownership,
request capacity and alignment, or opt into dynamic shared memory. The provider
remains authoritative for the required byte count and alignment.

A descriptor's `sharing` selects only the slice layout: `"shared"` overlaps
every call that passes the same descriptor on one region, while `"exclusive"`
gives each call site its own slice. A call site inside a loop reuses its slice
under either layout, so `auto_sync` is independent of `sharing` and defaults to
`False` for both.

The synchronization model is deliberately simple. A descriptor names one
region; distinct descriptors and compiler-owned storage never alias each other.
With `auto_sync=True`, the compiler appends
`cuda.syncthreads()` for block groups or `cuda.syncwarp(mask)` for Warp groups
immediately after every call that consumes the storage, including the last
one, and never inserts a barrier before a call. That trailing barrier only
orders reuse of the temporary storage; it is not a general barrier for the
kernel's own shared-memory traffic. With the default `auto_sync=False`, the
caller issues `cuda.syncthreads()` between consecutive uses, and a call site
inside a loop counts as a reuse on every iteration.
Compiler-owned storage always synchronizes.

All descriptors and compiler-owned requirements of a kernel share one
shared-memory backing. Above the 48 KiB static limit that backing moves to
dynamic shared memory and the launch reserves the exact byte count.
Supported Numba-CUDA-MLIR releases do not separate static and dynamic shared
allocations reliably. A kernel using cooperative temporary storage must not
also declare a zero-sized or runtime-sized `cuda.shared.array`. When
cooperative backing becomes dynamic, user static shared arrays are also
unsupported. Keep both user arrays and cooperative backing static, or move the
user data out of shared memory. Storage-free operations do not add this
restriction.

With `auto_sync=False`, a descriptor must originate from exactly one
constructor site. Selecting between multiple manual-sync constructors is an
MVP restriction: the compiler cannot prove that caller barriers protect the
merged region, even when a particular program supplies sufficient barriers.

Cooperative calls in device helpers must be inlined into the kernel; use
`@cuda.jit(device=True, inline="always")` when selecting the helper's
policy explicitly. Standalone collective helpers and collectives inside
standalone callbacks are unsupported. For the MVP, `literal_unroll`
values cannot determine cooperative payload extents, group dimensions,
selectors, or descriptor constructor arguments. Write separate calls with
explicit constants, or use an ordinary loop with one fixed cooperative shape.
An unrelated `literal_unroll` loop does not add this restriction.

Warp `transpose` uses compiler-owned storage with one disjoint slice per
physical or logical group and inserts `syncwarp` with the exact group mask.
Explicit `TempStorage` is rejected by both the common and qualified APIs for
every Warp Load and Store algorithm, including the storage-free modes.

These APIs are compile-time kernel constructs. Calling them outside a
compatible compiler context reports a structured context error.

See the [CCCL documentation](https://nvidia.github.io/cccl/unstable/python/coop.html)
for the complete signatures.
