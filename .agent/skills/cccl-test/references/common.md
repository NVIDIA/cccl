# Common CCCL Test Guidance

Apply this guidance across CCCL tests unless a path-specific test reference says otherwise.

## Local Consistency

- Read nearby tests first and mirror their directory layout, file names, helper types, includes, assertion style, and local gating or skip mechanisms.

## Coverage

- Cover relevant edge cases.
- Cover relevant input and output types.
- Cover error behavior when applicable.
- Cover runtime and compile-time behavior when applicable.
- Cover device and host behavior when applicable.

## Test Structure

- All tests must have the correct license banner.
- Use the local test harness assertions and helpers.
- Use compile-time checks for compile-time guarantees and constexpr coverage when relevant.
- Negative tests should check the intended diagnostic or failure mode when the local harness supports it.

## Portability

- Prefer project test macros and helpers for compiler, dialect, exception, host/device, and platform probes instead of spelling ad hoc checks directly.
- If a test is unsupported, expected to fail, disabled, or skipped on a platform, motivate it with a comment.

## Validation

- Use targeted test runs for the project and files being changed.
