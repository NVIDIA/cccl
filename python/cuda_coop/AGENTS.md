# `cuda.coop` Python guidance

- Organize primitives as vertical family slices across the portable API,
  CUTLASS, and Numba-CUDA-MLIR. Mirror semantic filenames where useful; do not
  add empty backend mirrors.
- In stubs, use `Name as Name` only for an intentional re-export. Use grouped,
  unaliased imports for annotation dependencies.
- In underscore-private modules, import public types directly; use descriptive
  non-underscored aliases only for type-name conflicts.
- Keep runtime declarations, stubs, documentation, and consumer typing tests in
  sync. Runtime compiler markers should remain import-light.
- Keep `_core` semantics and their documentation backend-neutral. Name a DSL
  there only in integration code that must know it; backends must not import
  one another.
