# `cuda.coop` Python guidance

- Organize primitives as vertical family slices across the portable API,
  CUTLASS, and Numba-CUDA-MLIR. Mirror semantic filenames where useful; do not
  add empty backend mirrors.
- In stubs, use `Name as Name` only for an intentional re-export. Use grouped,
  unaliased imports for annotation dependencies.
- Do not rename a public imported type to `_Type` merely to mark local use. An
  underscore-private module already marks its boundary; use leading underscores
  for local helpers and real name conflicts.
- Keep runtime declarations, stubs, documentation, and consumer typing tests in
  sync. Runtime compiler markers should remain import-light.
- Keep shared semantics in `_core`; backend modules must not import one another.
