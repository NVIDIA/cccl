# Rust `cuda.coop` guidance

Read the repository's root guidance first. Apply this file to the Rust
workspace, its examples and tests, and `docs/rust/`.

## Design tenets

The design tenets of `cuda.coop` are **performant, consistent, delightful**.

- **Performant:** target negligible overhead over CUB through compile-time
  planning, inlining and LTO. Avoid hidden runtime work. Inspect the final
  device artifact and measure representative execution before making claims
  about register use, spills or overhead.
- **Consistent:** preserve common operation names, group-first arguments,
  defaults, CUB behavior, mutation and result ownership. Document necessary
  Rust syntax and safety adaptations. Additional capabilities extend shared
  behavior without changing it.
- **Delightful:** use idiomatic Rust, useful defaults, inferred element types,
  reusable generic kernels, ordinary Cargo builds and actionable errors.

## API and examples

- Use the single `cuda_coop` API, normally with `use cuda_coop as coop;`.
  State the supported Oxide compiler/toolchain explicitly. Do not duplicate
  primitive namespaces to separate built-in operators from static callables.
- Make the main per-thread item count a kernel const generic named
  `ITEMS_PER_THREAD`, propagated through helpers and host specialization.
  Keep operation-specific auxiliary and output extents independent.
- Infer payload element types from typed producers or assignments while
  preserving widths. Investigate inference failures instead of adding a
  redundant element type to an ordinary example. Document full signatures.
- Load fills its payload in place and returns unit. Preserve fresh results
  where the common operation requires them, and reflect Store mutation in
  borrowing and validity. Undefined, partial and root-only results must not
  become unrestricted initialized Rust values.
- Preserve hierarchy units, supported groups, storage sharing and explicit
  synchronization obligations. A group descriptor or Rust borrow alone is
  not a proof of collective participation or shared-memory safety.
- Required collective completion must not depend solely on `Drop` running.

## Ownership and validation

Keep CUB specialization, exact scratch layout queries and operation semantics
in the native host provider. Keep compiler discovery, lowering, allocation,
device linking and launch enforcement in Oxide. Host compiler libraries and
filesystem dependencies must stay out of device code. CMake consumers do not
acquire a Rust requirement.

Every public item needs useful rustdoc: arguments and generic parameters,
defaults, supported types/groups, participation, layout, mutation and result
validity, scratch/synchronization, errors and examples. Every public unsafe
item needs a concrete `# Safety` contract. Use rustdoc as the API reference
and the CCCL Rust guide for complete programs and explanations.

Prefer end-user primitive tests, with item counts 1 and 4 where supported.
Use focused compiler/provider fixtures for independently reviewable compiler
capabilities. Keep displayed examples and their executed source identical.
Distinguish host tests, rustdoc/doctests, device compilation, GPU execution,
sanitizers and current-head CI. A sanitizer pass is not a Rust soundness proof.

Run formatting, applicable lints, warning-free documentation and meaningful
targeted tests. Correct foundational defects at their owning layer and
requalify dependents instead of hiding a redesign in a later primitive family.
