# `cuda.coop` architecture

This document explains how the portable API and compiler backends divide
responsibility. The directory layout follows primitive semantics so a developer
looking for `scan`, for example, follows the same path through the portable
core, CUTLASS, and Numba-CUDA-MLIR.

## Scope

The initial alpha provides the group-first primitive families documented in
the package README, a backend-neutral contract, and installed-wheel adapters
for CUTLASS Python DSL and Numba-CUDA-MLIR. Backend-qualified controls may
differ when the compilers or underlying CCCL providers differ. Device-wide
algorithms, a hidden workspace for grid collectives, and artificial parity for
backend-only controls are outside this release.

## Compilation pipeline

The portable API captures a compiler-independent operation. The active
backend then resolves compiler types and launch facts, lowers that semantic
operation to a CCCL provider, and incorporates the resulting code and scratch
requirements into the kernel:

```text
cuda.coop public call
    -> _core/api/<family>.py
    -> _core/group/<family>.py
    -> <backend>/_group_<family>.py
    -> <backend>/_lowering/_<family>.py
         |-> _core/{block,warp}/<family>.py  (shared CUB specs)
         `-> <backend>/_compiler/*
```

The stages have deliberately narrow ownership:

- `_core/api` owns portable signatures, argument validation, and backend
  delegation. It does not know how either compiler represents values.
- `_core/group` owns group semantics, participation contracts, scratch
  requirements, and stable semantic/artifact keys. It does not render or
  compile provider code.
- `_core/reduce.py`, `_core/scan.py`, and `_core/thread_group.py` own semantic
  models shared across physical scopes. `_core/block` and `_core/warp` own
  backend-neutral CUB provider specifications used by both lowerings; they are
  not additional public APIs.
- A backend `_group_<family>` module owns its qualified public signature and
  compiler-facing argument capture. It does not own generated code.
- A backend `_lowering/_<family>` module resolves compiler types, applies
  backend-only restrictions, and selects the CUB or CUDAX implementation. CUB
  and CUDAX are provider choices, not navigation layers.
- A backend `_compiler` package owns activation, compilation lifecycle,
  caching, storage materialization, and final linking. It does not define
  primitive-family semantics. Numba's compiler package also has
  `_group_<family>` planning and `_rewrite_<family>` finalization mixins because
  its early IR and post-inference stages see different information.

This boundary keeps semantic navigation consistent without pretending the two
compilers have the same execution model.

## Backend lifecycles

### CUTLASS Python DSL

CUTLASS traces qualified calls and registers immutable provider requests in a
per-trace session. Finalization renders all requested providers in canonical
order, resolves an exact AOT or cache hit when possible, otherwise compiles one
bundle, materializes deferred scratch storage, and attaches the resulting
linkable artifact to the kernel.

The ordering is a compatibility contract. An exact AOT hit must avoid header
discovery and mutable cache/toolchain state. A JIT path must preload NVRTC
before querying its version or constructing PCH state. Renderer kinds, source
bytes, generated symbols, bundle identities, cache schemas, and AOT manifests
must not change merely because Python modules move.

AOT portability is governed by the provider ABI plus exact rendered source,
bundle format, architecture, compiler options, layout expressions, and linker
compatibility. The writer version is diagnostic rather than an equality gate,
and current headers are intentionally absent from an exact-hit lookup. A
header or provider change that can alter the ABI or semantics of identical
rendered source therefore requires a provider-ABI bump.

### Numba-CUDA-MLIR

Numba-CUDA-MLIR performs cooperative planning before type inference. Its
registered planners recognize exact `cuda.coop` callables, reconcile explicit
and inferred launch dimensions, and defer device-function decisions to the
eventual kernel caller. Its single-phase rewrite is also registered before
inference; it resolves payload dtypes from IR definitions and tracked
provenance, materializes provider code and scratch storage, and records link
artifacts.

The split preserves information that is unavailable at the initial IR rewrite.
Launch precedence, device-function deferral, dtype precedence, callable
identity hashing, and cache keys are semantic contracts; moving code between
modules must not alter them.

Both adapters register at qualified import time because their compilers must
recognize the exact callable objects before kernel analysis begins. Registered
hooks use exact identity and are inert for unrelated functions. Applications
that manage activation can disable portable-root probing before import with
`CUDA_COOP_DISABLE_AUTO_DSL_REGISTRATION=1`.

## Shared concepts

- `ThreadData` is a fixed-size per-thread register payload. Operations return
  fresh results unless their public contract explicitly says otherwise.
- `TempStorage` represents caller-provided or compiler-planned scratch space.
  Portable planning describes capacity and alignment; each compiler owns its
  final storage representation.
- `ThreadGroup` describes physical or logical participation. Launch facts are
  compiler evidence used to resolve that description, not part of the public
  container itself.
- Semantic and artifact keys describe behavior and generated-provider
  identity. Persisted cache or AOT formats are versioned only when their actual
  representation changes; directory or module renames do not justify a new
  version.

## Adding a primitive family

Add a family as a vertical slice rather than to a catch-all dispatcher:

1. Define the portable call and stub in `_core/api/<family>.py[i]`.
2. Define group semantics and planning in `_core/group/<family>.py` with
   characterization for semantic and artifact keys.
3. Add matching `_group_<family>.py[i]` and `_lowering/_<family>.py` modules in
   each supported backend. Add Numba `_compiler/_group_<family>.py` or
   `_rewrite_<family>.py` mixins when that family needs early planning or
   post-inference finalization; do not add empty mirror modules.
4. Register the exact qualified callable identity with the backend compiler.
5. Add semantic-family tests at the appropriate evidence level: `unit` for
   pure behavior, `compile` for planning/lowering/generated artifacts, and
   `runtime` for launched GPU semantics.
6. Update portable re-exports, installed-wheel typing coverage, examples, or
   public documentation when the public surface changes.
Backend-specific compiler tests should remain backend-specific. Matching
semantic filenames help navigation; empty mirror modules or tests do not.
