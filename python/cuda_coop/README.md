# cuda-coop

`cuda-coop` is the standalone Python distribution for cooperative primitives
shared across CUDA Python DSL backends. The unqualified import exposes the
portable common v1 group-first API:

```python
from cuda import coop
```

Importing the root safely probes the installed CUTLASS and Numba-CUDA-MLIR
runtimes, checks the concrete compiler capabilities `cuda.coop` needs, and
registers each compatible backend. Missing runtimes are ignored. An installed
but incompatible runtime produces a
`CudaCoopAutoRegistrationWarning` with the missing capability and upgrade
guidance while leaving the common API usable. Set
`CUDA_COOP_DISABLE_AUTO_DSL_REGISTRATION=1` before import to disable these
probes.

The v1 profile includes group and storage objects; Load and Store; Reduce and
Sum; Scan aliases; Exchange and Shuffle; Adjacent Difference and single-output
Discontinuity; key and key/value forms of Merge Sort, Radix Sort, and TopK;
keys-only Radix Rank; Histogram; and Run-Length Decode. Promotion requires
matching semantics and result layout in CUTLASS and
Numba-CUDA-MLIR. Backend callbacks, side outputs, native payload forms, and
routing controls remain qualified.

The common numeric family is the exact CUTLASS/Numba intersection: `uint8`,
`int32`, `uint32`, `int64`, `uint64`, `float32`, and `float64`. Python `int`
and `float` select `int32` and `float32`. Integer-key and counter operations
use the narrower families documented with those operations. Complex and
extension aggregate collectives remain qualified Numba-CUDA-MLIR features;
passing a qualified payload to an unqualified operation does not widen this
contract. Qualified CUTLASS `ThreadData[T]` is a generic register container,
so it can preserve complex or user-defined values for indexing and backend
handoff, but CUTLASS collectives still accept only the dtype families their
providers implement.
Python's static type system treats `bool` as a subtype of `int`, so an editor
cannot express that one exclusion precisely; compiler-time validation still
rejects boolean payloads from the common numeric family.

Common Shuffle is the exact overlapping block form: a fixed-size per-thread
payload, `mode="up"` or `mode="down"`, and `distance=1`. It leaves the vacated
first or last flattened item undefined. Scalar modes, other distances, and
boundary outputs remain backend-qualified.

Common Histogram accepts a complete block and a fixed-size `ThreadData`
samples payload without mutating it. `bins` and `bins_per_thread` are positive
trace-time integers, and `bins` must not exceed
`group_size * bins_per_thread`. The returned `ThreadData` is striped: member
rank `r` owns bins `r + i * group_size`, and positions beyond `bins` are zero.
Every sample must satisfy `0 <= sample < bins`. Portable V1 certifies
`uint8`, `int32`, `uint32`, `int64`, and `uint64` samples and `int32`,
`uint32`, `int64`, and `uint64` counters; other dtype support is
backend-qualified. The Python `int` dtype spelling maps to `int32`.

Backend-qualified packages remain available for backend-specific extensions:

- `cuda.coop.cutlass`
- `cuda.coop.numba_mlir`

### Typing and editor completion

The wheel ships root and backend-local `py.typed` markers plus public stubs.
The marker files are intentionally empty: under PEP 561 their presence marks a
package as typed, while the substantive declarations and docstrings live in
the adjacent `.pyi` files. `ThreadData` and `TempStorage` are
compiler-dispatched factories; use the exported `ThreadDataLike[T]` and
`TempStorageLike` protocols when annotating backend-neutral helpers. An editor
shows the portable common contract for `from cuda import coop` because compiler
selection cannot change a Python module's static type. Compiler activation
changes lowering, not the IDE's static view. Use a qualified import for
DSL-specific signatures, docstrings, and code completion:

The common runtime also recognizes selected built-in callable objects such as
`operator.add` by identity. Python annotations cannot describe the identity of
one callable without accepting arbitrary callbacks, so the common stubs expose
only the equivalent string literals (`"add"`, `"multiply"`, and so on). Use
those spellings for strictly typed portable code; qualified APIs type custom
callbacks where the backend supports them. CUTLASS group-first calls have the
same identity limitation and expose supported literal spellings in their
stubs. Numba-CUDA-MLIR group-first calls do the same; its qualified root accepts
custom callbacks where the backend supports them.

```python
import cuda.coop.cutlass as coop
# or
import cuda.coop.numba_mlir as coop
```

CUTLASS and Numba-CUDA-MLIR qualified imports provide the activation fallback
used when automatic probing is disabled. A CUTLASS import validates that its
runtime is importable and installs the qualified root fallback, while
Numba-CUDA-MLIR validates and registers its planner and rewrite hooks
transactionally. The automatic root probe additionally checks the concrete
CUTLASS compiler features required by generated providers.

Python's standard warning controls can filter registration diagnostics. For
example, `PYTHONWARNINGS='ignore:cuda.coop automatic DSL registration'`
suppresses only these warnings; disabling auto-registration is preferable when
an application intentionally manages compiler activation itself.

The wheel contains the common API plus the CUTLASS and Numba-CUDA-MLIR
adapters and stubs. Extras add compiler dependencies.
The plain `cuda-coop` does not install a compiler. `cu12` and `cu13` add
toolkit dependencies. The `cutlass` and `cutlass-cu13` extras select CUDA 13
and `nvidia-cutlass-dsl>=4.8,<5`; `cutlass-cu12` selects the same compiler
range with CUDA 12 dependencies. The `numba-cuda-mlir` and
`numba-cuda-mlir-cu13` extras select CUDA 13 and
`numba-cuda-mlir[cu13]>=0.5.0`; `numba-cuda-mlir-cu12` selects the CUDA 12
variant. The explicit spellings `cutlass-cu12`, `cutlass-cu13`,
`numba-cuda-mlir-cu12`, and `numba-cuda-mlir-cu13` remain available.

## Initial Boundaries

- `cuda.coop.cutlass` is the shared CUTLASS cooperative-primitive surface and
  is common-v1 conforming.
- `cuda.coop.numba_mlir` is the qualified cooperative-primitive frontend for
  `numba-cuda-mlir` kernels and is common-v1 conforming.
  It combines the CuTe provider implementation and Prims array adapter behind
  one public root API.

## Layout

```text
python/cuda_coop/
├── CMakeLists.txt
├── pyproject.toml
├── cuda/
│   └── coop/
│       ├── __init__.py
│       ├── _headers/
│       ├── _core/
│       ├── cutlass/
│       └── numba_mlir/
└── tests/
    ├── contracts/
    ├── backends/
    ├── providers/
    ├── integration/
    └── support/
```

The `cuda-coop` distribution owns `cuda/coop/__init__.py` and the backend
packages under `cuda.coop.<dsl>` added by their commits. The wheel bundles its own private
libcudacxx, CUB, Thrust, and CUDAX headers under `cuda.coop._headers`; no
separate header distribution is required. The top-level `cuda` directory
remains a namespace package without `cuda/__init__.py`.

## Backend-neutral primitive semantics

`cuda.coop._core` is an internal, standard-library-only semantic layer shared
by backend frontends. It describes normalized cooperative primitive
specializations and group-lowering plans; backends continue to own validation,
compilation, linking, caching, rewriting, provider requests, and launching.
BlockTopK, BlockScan, BlockExchange, WarpScan, WarpExchange, and group Reduce,
Scan, and Exchange use these shared contracts. CUTLASS group Reduce selects
broadcasted CUDAX for the default full-group route and exact CUB block/warp
specializations for supported CUB-only variants. See
[`docs/core-primitive-architecture.md`](docs/core-primitive-architecture.md)
for the dependency rules, adapter contract, and validation coverage.

## CUTLASS Backend

Use the qualified package for direct CUTLASS code and CUTLASS-specific editor
completion:

```python
import cuda.coop.cutlass as coop
```

It exposes `ThreadData`, `TempStorage`, group descriptors and `this_*`
helpers, plus the group-first collectives as they land on the portable
contract, beginning with `load` and `store`. Importing
`from cuda import coop` performs this activation automatically when the
installed CUTLASS runtime satisfies the required capability contract. The
qualified import remains useful for CUTLASS-specific signatures and helpers.

Load and Store accept compiler-traced tensors and CUTLASS array ("Prims")
payloads behind one root API, with partial-tile `valid_items`/`oob_default`
controls and element offsets. Wrappers are rendered once per trace, compiled
to LTO-IR with NVRTC against the wheel's private CCCL headers, and attached
during CUTLASS finalization.

### Reduce result ownership

Full physical block and warp results are defined on every member, so
rank-zero-only consumption remains valid but is not required. For a valid-count
block or physical-warp reduction, fold each member's items first and consume the
direct-CUB scalar only at group rank zero. This contract deliberately does not promise
bitwise compatibility with the former floating-point reduction tree. Unsigned
integer arithmetic follows the selected type's modular C++ semantics; inputs
that would cause signed integer overflow are outside the supported contract.
Floating-point sum/product and min/max follow the selected official CUDAX/CUB
implementation, so reassociation may change rounding, NaN
selection/propagation, and signed zero. Tests require common and qualified group
calls that select the same artifact to be bitwise identical.
The selected official CUDAX/CUB route is also normative for NaN payload and
selection, infinities, and signed zero; direct Python and direct C++
probes for one route must match bitwise on one toolchain. Finite mathematical
sum/product results use `rtol=2e-6, atol=2e-6` for `float32` and
`rtol=1e-12, atol=1e-12` for `float64`. No equivalence is promised across
different routes or algorithms.

The group-first route contract is:

| Scope and request | Official route | Result contract |
| --- | --- | --- |
| Full block or physical warp, scalar or `ThreadData` | broadcasted CUDAX Reduce | scalar, defined on every member |
| Full thread, cluster, or static `group_by` group | CUDAX Reduce | scalar, broadcast to every member by default or root-owned with `broadcast=False` |
| Grid group | blocked | waits for a reviewed compiler-managed device-workspace contract |
| Group-first full block or physical warp with `broadcast=False` | root-only CUDAX Reduce | scalar, defined only at group rank zero |
| Any supplied valid count on block or physical warp, scalar input | direct CUB Reduce | scalar, defined only at group rank zero, including when the count equals the group size |
| Any supplied valid count on block or physical warp, `ThreadData` input | unsupported | explicitly fold lane items to a scalar first |
| Full block with explicit CUB algorithm | direct CUB `BlockReduce` | scalar, defined only at block rank zero |

Focused planner, runtime, and final-cubin tests cover the direct-CUB Reduce
selectors.

The common-v1 exact conformance cohort exercises non-sum `reduce` with `max`
on both complete blocks and physical warps under CUTLASS and
Numba-CUDA-MLIR. Its explicit block-algorithm and warp-valid-count routes use
`broadcast=False`, consume the result only at group rank zero, compare the
common and qualified spellings with an independent oracle, preserve the input,
and require the linked provider wrappers to disappear from the final binary.


### Group-first Scan and Exchange

Scan and Exchange use the same compile-time group dispatch as Reduce:

```python
group = coop.this_block()
prefix = coop.scan(group, thread_items, mode="exclusive")
striped = coop.exchange(group, prefix, mode="blocked_to_striped")
```

`coop.scan` lowers block scalar or `ThreadData` operands to public
`cub::BlockScan` and physical-warp scalar operands to public `cub::WarpScan`.
`coop.exchange` accepts `ThreadData` for complete blocks or physical warps and
lowers `striped_to_blocked` and `blocked_to_striped` to public
`cub::BlockExchange` or `cub::WarpExchange`. Exact launch dimensions select the
CUB specialization; an upper bound such as `maxntid` is never treated as an
exact shape. Block Exchange supports one through five items per thread;
logical/scatter WarpExchange supports up to four. Guarded block scatter
requires signed `Int32` or `Int64`
ranks because rank -1 is its no-write sentinel; the other block
scatter modes retain all supported integral rank dtypes. The group marker
affects planning and artifact identity but is erased before the runtime FFI
ABI.

Scatter ranks are a caller precondition: every participating write rank
must address its block or logical-warp tile and write ranks must be unique.
For flagged block scatter, only ranks with a true flag participate; guarded
block scatter accepts rank -1 as the sole no-write sentinel. Duplicate write
ranks and destinations without a writer have unspecified values. The
direct-CUB migration no longer preserves the retired block shim's behavior for
invalid ranks. Logical/scatter WarpExchange uses `Int32` ranks and requires the
exact CTA thread count to be divisible by `threads_in_warp`.

Full-block, physical-warp, and logical-warp group calls delegate through the
same planners and typed provider artifacts. Exchange wrappers make one
whole-register-array CUB call and normally own exact typed scratch. The
compiler-planned storage described below makes size-less storage operational
for block Load, Store, Scan, Adjacent Difference, Discontinuity, Radix Sort,
Merge Sort, and BlockExchange.

No handwritten Scan or Exchange collective remains as an automatic fallback.

### Provider compilation and caching

Provider requests, generated source, features, includes, and symbols are
canonicalized before compilation. The persistent provider cache validates
non-editable wheel headers from complete hashed PEP 376 `RECORD` entries, then
uses Git tree identities for remaining roots in clean source checkouts.
Dirty Git trees and roots without usable wheel or clean-Git provenance use a
conservative content walk. Cache artifacts and their metadata are published
atomically under per-artifact locks. Cache schema v3 deliberately ignores
artifacts written by older schemas, so the first provider resolution after the
upgrade is cold and later exact resolutions reuse v3 entries.

The PEP 376 fast path treats a non-editable wheel installation as immutable; it
is an installation identity, not a live integrity scan of every header.
Reinstall a wheel instead of editing its installed headers in place. Use a
clean Git or custom header root while developing header changes.


## Numba-CUDA-MLIR Backend

The portable group-first spelling next to `numba_cuda_mlir.cuda` is:

```python
from numba_cuda_mlir import cuda
from cuda import coop
```

The bare root capability-checks an installed Numba-CUDA-MLIR runtime and
registers its planner and rewrite before compilation. The qualified import does
the same explicitly when automatic registration is disabled. Its
whole-function planner recognizes common-root object identities and lowers them
through the normal Numba-CUDA-MLIR providers. Use
`import cuda.coop.numba_mlir as coop` for backend-specific operations and
Numba-CUDA-MLIR-specific editor completion.

A portable kernel body is:

```python
@cuda.jit
def kernel(source, destination, totals):
    tid = cuda.threadIdx.x
    block = coop.this_block()
    items = coop.ThreadData(1)
    coop.load(block, source, items)
    coop.store(block, destination, items)
    totals[tid] = coop.reduce(block, items[0])
```

The qualified package shares CUTLASS's thread hierarchy vocabulary:
`ThreadHierarchy`, `ThreadGroup`, `this_thread`, `this_warp`, `this_block`,
`this_cluster`, and `this_grid`. It also exports the common-v1 root
operations as they land on the portable contract, with the same positional
and keyword order.

`rank`, `count`, `rank_as`, `count_as`, `sync`, `sync_aligned`, and
`is_member` lower when the configured launch provides their required facts.
Static `group_by` mappings partition threads within a warp or complete warps
within a block.

The unqualified root contains the broad common-v1 group-first profile. The
qualified Numba-CUDA-MLIR surface additionally exposes backend algorithm
enums through the same group-first interface:

- Root Load and Store lower complete physical blocks and warps through the
  established public CUB implementations.
- Root Reduce lowers full thread, physical-warp, block, SM90+ cluster, and
  static mapped groups through CUDAX.
- A root Reduce with `valid_items`, or with a block `algorithm`, lowers through
  direct CUB and requires `broadcast=False`.
- Grid rank and count use exact configured launch dimensions. Grid sync is
  rejected because the current launcher cannot request a verified cooperative
  launch, and Grid Reduce is rejected until a reviewed compiler-managed
  device-workspace contract is available.

Group planning runs after device-function inlining. It detects group markers
before promoting exact launch facts into the current compiler state, so typing
and lowering continue in the same attempt without replaying planners merely to
activate launch facts. Generated CUDAX and CUB providers attach as real
LTO-IR, and their device overloads are forced inline so the final cubin does
not retain provider call frames.

The runnable group example is
[`examples/numba_mlir/group_hierarchy.py`](examples/numba_mlir/group_hierarchy.py).

This experimental path requires the whole-function planner and configured
launch contracts proposed in upstream `numba-cuda-mlir`
[#202](https://github.com/NVIDIA/numba-cuda-mlir/pull/202) and
[#203](https://github.com/NVIDIA/numba-cuda-mlir/pull/203). The complete
validated source stack also includes
[#238](https://github.com/NVIDIA/numba-cuda-mlir/pull/238) for planner
dynamic-shared-memory minima and
[#239](https://github.com/NVIDIA/numba-cuda-mlir/pull/239) for scalar-literal
specialization retry. Until releases contain those contracts, use a compatible
source build.

## Local Validation

From the repository root:

```bash
python -m pytest -q -p no:cacheprovider python/cuda_coop/tests
python -m ruff check python/cuda_coop
python -m compileall -q python/cuda_coop/cuda/coop
```

The Numba-CUDA-MLIR corpus lives under `tests/backends/numba_mlir`. Pure host
checks, compiler checks, representative GPU runtime coverage, and broad stress
matrices have separate directories. Use the focused layers during iteration
and reserve `stress` for scheduled or final qualification.
