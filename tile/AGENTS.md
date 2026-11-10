# CCCL Tile guidance

This subproject provides cooperative primitives for Tile programs. Tile-mode
support within libcu++ and CUB follows their guidance and
`docs/libcudacxx/tile.rst`.

Apply this guidance to the native implementation, language frontends, tests,
documentation, and runnable examples. Follow the repository's root guidance and
applicable style and test skills.

## Design tenets

- **Performant:** target negligible overhead over the underlying primitives
  through compile-time specialization, inlining, and LTO. Avoid hidden
  allocations, transfers, host or stream synchronization, and kernel launches.
- **Consistent:** align operation names, mathematical behavior, options, and
  result ownership across languages. Preserve and document intentional
  differences in syntax, shapes, and type behavior.
- **Delightful:** use idiomatic interfaces, useful defaults, and shape and type
  inference. Diagnose unsupported uses clearly.

## Tile semantics

APIs consume logical Tile values and return results. Preserve input values,
documented result shapes, element types, and key/value association. Make each
operation's constraints and valid output regions explicit.

The compiler selects physical participation and allocates temporary storage.
Keep per-thread storage and scratch-management concepts out of ordinary Tile
calls. Expose topology only when it changes an operation's meaning, such as an
exchange layout. Use each language's own load, store, launch, and composition
facilities. Tile operations own their required internal synchronization;
callers must not need extra barriers around a collective.

Preserve CUB's supported numerical and ordering behavior. Do not promise stronger
determinism, tie ordering, overflow behavior, or output validity. Document
frontend adaptations such as reduction axes, broadcasting, and dtype promotion;
cross-language consistency must not silently change established contracts.

## Ownership and interfaces

The shared provider and TileLib belong under `tile/cpp/tilelib`, public C++
headers under `tile/cpp/include`, and Python sources under
`python/cuda_coop_tile`. Language frontends share one native implementation.
C++ consumers must not require a Python package installation. Python packaging
consumes the native build's artifacts.
Keep build-tree and installed CMake usage requirements equivalent, including
compiler modes and host-compiler options.

The TileLib contains the provider that selects and builds native implementations.
The canonical interface is `tile/cpp/tilelib/interface/cuda_coop_tile.mlir`.
Edit it, regenerate declarations and schemas with
`tile/tools/generate_tile_declarations.py`, and verify agreement. Preserve
declaration, ABI, and artifact identity checks. Invalid explicit inputs must
produce useful errors rather than select an unrelated fallback.

State build prerequisites accurately. Preview SDK inputs supplied separately
must remain outside source and package payloads until approved for distribution.
When a native build is requested, missing SDK inputs must fail configuration
with a diagnostic naming the missing input.
Keep internal provenance and qualification records in private ledgers.

## Tests and documentation

Favor public primitive and realistic composition tests over tests that mirror
implementation details. Cover supported shapes, dtypes, ownership, meaningful
edge cases, and diagnostics. Exercise multiple logical extents and supported
multidimensional cases; vary the compiler's worker-count hints when testing
participation behavior.
When testing input preservation, store the original logical Tile after the final
collective and compare it with an independent expected value.

Shared native changes need relevant Python and C++ coverage. Validate installed
artifacts and standalone C++ consumption when builds, packaging, headers, or
discovery change. Keep failures visible; distinguish local, CI, and release
qualification.

Verify that Python tests exercise the intended checkout or installed artifact,
including with a competing editable install. Use an actual backend-generated
artifact for that check. Restore test doubles without breaking the identity of
loaded extension modules.

Keep API docstrings, typing declarations, C++ declarations, references, and
examples consistent. Tile documentation lives under `docs/tile`, with Python
and C++ sections linking to shared concepts. Update literalinclude sources and
execute affected examples. Inspect rendered navigation and references after
documentation moves.
