# `cuda.coop` agent guidance

Apply these conventions to package code, README and guide snippets, API
docstrings, runnable examples, and test snippets included in documentation.

- Write the API/project name as `cuda.coop` and the distribution as `cuda-coop`.
- In every public `ThreadData` example, pass the item count explicitly:
  `coop.ThreadData(items_per_thread=2)`.
- Omit the constructor's element-type argument, whether positional or named.
  Apply this to common and backend-qualified APIs and aliases of `ThreadData`.
- Use a supported typed producer or typed assignments to supply the element
  type. Preserve intentional widths. If inference fails, investigate the
  backend limitation or revise the example; do not silently add a constructor
  type argument.
- Keep formal API signatures accurate. Tests specifically checking explicit
  types, invalid arguments, or positional compatibility may exercise those
  forms; do not use them as public examples.
- Update the source of `literalinclude` snippets and run affected examples
  when changing them. Apply changes at the first owning PR and preserve them
  when restacking both the Numba-CUDA-MLIR and CUTLASS integrations.
