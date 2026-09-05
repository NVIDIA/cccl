<!--
Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.

SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# cuda.coop Test Instructions

These instructions apply throughout this test tree and supplement the repository
`AGENTS.md`. Before changing tests, read the [suite README](README.md), the
[coverage manifest](contracts/coverage.toml), the repository `cccl-test` skill,
and nearby tests.
Keep inventories, current test counts, dependency versions, and CI availability
in their canonical configuration or generated reports. Do not copy them here.

## Place Tests by Ownership

- Put backend-free semantics, normalization, API surfaces, and parity obligations
  in `contracts/`.
- Put frontend, lowering, compilation, and representative execution under
  `backends/<backend>/<evidence>/`.
- Put provider ABI, source generation, metadata, and final-link checks in
  `providers/`.
- Put broad Cartesian, randomized differential, architecture, SASS, resource,
  and stress coverage in qualification or stress suites.
- Put examples, compiler, and packaging behavior in `integration/`.
- Put reusable cases, oracles, fixtures, and toolchain helpers in `support/`;
  support modules must not become independently collected suites.

Each test module owns one backend, one primitive family, and one evidence layer.
Split a module when its path can no longer communicate those three facts.

## Formulate Evidence Precisely

- Update `COMMON_PROFILE_MATRIX` when a common V1 capability or obligation
  changes. Update `contracts/coverage.toml` for qualified surfaces and policy.
- An `evidence_for` marker records only what that test directly proves.
  Multiple tests may support the same evidence cell. Compilation is not runtime
  evidence; source generation is not final-link evidence. A marker declares
  intent; only a fully passing logical test earns the evidence.
- Do not mark a migrated capability `required` until every required evidence
  cell has a real claim and an owning CI selection. New capabilities should not
  add migration debt.
- Exhaust cheap enums, boundaries, validation, planning, and diagnostics in
  host contracts. Compile once per distinct route and run one canonical case
  per genuinely different semantic or ABI branch.
- Move matrices larger than the documented pull-request limit to qualification
  unless each case has a recorded route distinction and an approved waiver.
- Assert behavior, diagnostics, or artifacts. Inspect exact source text only
  when source form is intentionally part of the public contract.
- Do not inspect external mutable registries or private global state to assert
  registration, ordering, or capability. Verify those integrations through
  documented public APIs and observable compile or runtime behavior. Isolate
  and explain unavoidable low-level compiler or IR dependencies in
  backend-owned support modules.
- Give regression cases an issue or bug reference and preserve the smallest
  input that demonstrates the failure.

## Preserve Collection Isolation

- At module import time, do not import optional backends, compile code, inspect
  a GPU, mutate process-global environment, or change `sys.path`.
- Import optional stacks inside tests or fixtures and use shared discovery,
  environment, compiler-cache, synchronization, and artifact fixtures.
- Use the deterministic RNG fixtures. Seed any framework-local generator from
  `deterministic_seed`; never rely on ambient random state.
- Run GPU, link-sensitive, large, stress, and qualification work serially unless
  the runner explicitly assigns independent GPUs. Do not add unbounded
  `-n auto`.

## Make Outcomes Deliberate

- Negative tests check the intended exception and exact stable diagnostic.
- An xfail must be conditional, strict, issue-linked, and carry an expiry date.
  Do not turn broad exceptions or numerical mismatches into xfails.
- Explain platform skips at the test site. Required backend lanes must fail
  their prerequisite preflight rather than pass through missing-stack skips.

## Validate and Hand Off

- Run the narrowest affected tests, then the relevant contract/parity audit and
  minimal-environment collection check described in the suite README.
- If the manifest changes, regenerate its human-readable coverage page and
  inspect the migration-debt report.
- Run pre-commit on every changed file. If CI selection changes, also exercise
  the affected driver dry-runs and change-detection fixtures.
