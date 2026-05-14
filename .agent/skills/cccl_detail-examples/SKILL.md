---
description: |
  Top-level `examples/` directory — CPM-consumption tests that verify CCCL can be
  fetched via CPM and built downstream. Covers `cccl_add_compile_test` signature,
  how each subdirectory is a self-contained CMake project, differences from
  per-library examples, the `packaging` CI preset, and authoring a new example.
  Triggers: "how do the top-level examples work", "add a new example", "what is
  cccl_add_compile_test", "examples/ CPM test".
---

The top-level `examples/` directory is not a unit-test suite — it is a set of
downstream-consumer validation builds. Each subdirectory is an independent CMake
project that fetches CCCL via CPM and builds against it, proving that a real user
can depend on CCCL from GitHub.

## What the directory contains

| Subdirectory                | What it exercises                                       |
|-----------------------------|-------------------------------------------------------|
| `basic/`                    | Bare `CCCL::CCCL` link target; serves as the canonical starter template |
| `ccclrt/`                   | `ccclrt` kernel launch patterns                         |
| `cudax/`                    | `CCCL::cudax` (requires `CCCL_ENABLE_UNSTABLE ON`)      |
| `cudax_stf/`                | STF APIs                                                |
| `thrust_flexible_device_system/` | Thrust with configurable device system (CUDA/OMP/TBB/CPP) |

Each subdirectory carries its own `cmake/CPM.cmake`, `CMakeLists.txt`, and at
least one source file. They have no dependency on the CCCL build system and
compile cleanly as standalone projects.

## How `cccl_add_compile_test` works

Defined in `cmake/CCCLUtilities.cmake`. Signature:

```cmake
cccl_add_compile_test(
  <output_test_name_var>
  <name_prefix>       # e.g. cccl.example
  <subdir>            # relative path to the standalone project
  <test_id>           # disambiguates multiple configs for the same subdir
  [CTEST_COMMAND <cmd>]
  [<additional cmake -D options>...]
)
```

The function registers a CTest test whose command is `ctest --build-and-test`,
which configures, builds, and runs the subdirectory's own CTest suite in one
step. The resulting test name is `<name_prefix>.<subdir>.<test_id>`.

The top-level `examples/CMakeLists.txt` passes two CPM-overriding variables to
every invocation:

- `-DCCCL_REPOSITORY` — local repo path (overrides the public GitHub URL during CI)
- `-DCCCL_TAG` — `GITHUB_SHA` in CI, `HEAD` locally

This lets CI validate the current PR's tree without pushing to GitHub first.

## Per-library examples vs top-level `examples/`

Per-library examples (e.g., `cub/examples/`, `thrust/examples/`) live inside their
library's source tree, link against the in-tree build, and are controlled by each
library's own CMake options. They test the in-tree build.

Top-level `examples/` tests the *packaging and export* surface: does `CPMAddPackage`
produce a usable `CCCL::CCCL` target for a downstream CMake project?

## CI integration

Examples run under the `packaging` CMake preset (`CCCL_ENABLE_EXAMPLES=true`,
`CCCL_ENABLE_TESTING=true`). The driver script is `ci/test_packaging.sh`, which:

1. Sets `CCCL_EXAMPLE_CPM_REPOSITORY` to the local repo root.
2. Sets `CCCL_EXAMPLE_CPM_TAG` to `GITHUB_SHA` (or `HEAD` locally).
3. Configures and builds the `packaging` preset.
4. Runs `ctest`.

In `ci/matrix.yaml`, `project: packaging` entries appear in `pull_request`,
`pull_request_lite`, and `nightly` workflows, covering CTK 12.0–13.X with gcc
and clang. A `-min-cmake` variant tests against CMake 3.18 (the minimum required).
GPU is required — the examples run device kernels via CTest.

## Authoring a new example

1. Create `examples/<your-example>/` with:
   - `cmake/CPM.cmake` — copy from any existing subdirectory.
   - `CMakeLists.txt` — `project(...)`, `include(cmake/CPM.cmake)`, `CPMAddPackage(NAME CCCL ...)`, add target, `include(CTest)`, `add_test(...)`.
   - Source file(s).
2. Register in `examples/CMakeLists.txt` with one or more `cccl_add_compile_test` calls.
3. If the example needs a non-default Thrust device system or other per-config variation, add a `foreach` loop (see `thrust_flexible_device_system` for the pattern).
4. Verify locally: `cmake -S . -B build -DCCCL_ENABLE_EXAMPLES=ON && cmake --build build && ctest --test-dir build`.

## Additional resources

- `references/docs.md` — index of `examples/` documentation (README files for each example).
