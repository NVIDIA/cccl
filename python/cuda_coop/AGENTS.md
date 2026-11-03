# `cuda.coop` agent guidance

Apply this guidance to package code, tests, documentation, API docstrings, and
runnable examples.

# Design Tenets

As a guiding principle, the design tenets behind `cuda.coop` are
**performant, consistent, delightful**:

- **Performant:** target negligible overhead over the underlying primitives
  through compile-time planning, inlining, and LTO; avoid hidden work.
- **Consistent:** the same operation should look and behave the same across
  DSLs, including argument order and result ownership.
- **Delightful:** Pythonic APIs, useful defaults, and inference should make
  common cases work out of the box, with clear errors for unsupported uses.

# Conventions

Write the API/project name as `cuda.coop` and the distribution as `cuda-coop`.

Use these import conventions in each module:

- Import the common API with `from cuda import coop` and use
  `coop.<primitive>` in DSL kernels.
- If the module does not need the common API, import its qualified API as
  `coop`: `import cuda.coop.numba_mlir as coop` or
  `import cuda.coop.cutlass as coop`. Use `coop.<primitive>` in kernels.
- When the module needs both common and qualified APIs, keep
  `from cuda import coop` for the common API. Use
  `import cuda.coop.numba_mlir as numba_coop` and/or
  `import cuda.coop.cutlass as cutlass_coop` for the qualified APIs, and use
  their respective aliases in kernels.

Regarding `coop.ThreadData` (including qualified APIs and import aliases):

- Prefer an `items_per_thread` kernel parameter for the main per-thread
  payload size, so callers can reuse the kernel with different item counts.
  Carry it through host launchers and device helpers as needed. The value
  must be known during compilation: Numba-CUDA-MLIR specializes the kernel
  argument, and CUTLASS kernel and launcher parameters use
  `items_per_thread: cutlass.Constexpr`.
- When the argument is a variable named `items_per_thread`, prefer
  `coop.ThreadData(items_per_thread)`. Otherwise, use the explicit
  `items_per_thread=` kwarg, including for integer literals and expressions.
- Keep fixed-size auxiliary payloads at the sizes required by their
  operations; they need not match the kernel's main `items_per_thread`.
- Omit the `dtype=` kwarg and positional dtype arguments from examples.
- Infer the element type from supported producers or typed assignments,
  preserving intended widths. If inference fails, investigate or revise the
  example; do not silently add `dtype=`.
- Document the actual constructor signature, including its supported `dtype=`
  parameter. These example conventions do not change the accepted arguments.
- Tests specifically checking `dtype=`, positional arguments, or invalid inputs
  may keep those calls. Do not use those tests as public examples.

Treat `cuda.coop` as the common API contract. Qualified APIs such as
`cuda.coop.numba_mlir` and `cuda.coop.cutlass` must be otherwise identical
supersets built on that contract: the same names, argument order, defaults,
behavior, and result ownership for shared operations. Add only the extensions
needed by each DSL.

Ensure new primitives have thorough docstrings and are captured in the API
reference docs; see existing precedence in code base.

Preserve CUB behavior for all primitives and don't invent new or stronger
semantics than CUB provides.

Tests should be geared toward testing the end-user primitives, not internal
helper functions or supporting glue--that infrastructure is exercised and
implicitly tested when doing primitive-level tests.

Parametrize kernel tests over `items_per_thread` where the operation supports
multiple item counts; prefer `pytest.mark.parametrize("items_per_thread", [1, 4])`
as the default coverage. Derive input sizes, launch arguments, and expected
results from that value. Preserve operation-specific constraints and focused
coverage for literal constructor arguments and invalid inputs.

Update the source of `literalinclude` snippets and run affected examples when
changing them. Apply changes at the first owning PR and preserve them when doing
PR restacking.
