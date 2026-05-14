---
description: |
  CCCL C++ test paths — fast iteration first, full matrix when needed.
  Covers `ci/util/build_and_test_targets.sh` for targeted CTest/lit runs
  and `ci/test_*.sh` for full host/std/arch matrix tests. GPU required for test scripts.
  Triggers: "run test Y", "how do I run the cub tests", "test libcudacxx",
  "full matrix test", "run just one test".
---

# cccl-test

Two test paths. Start with the targeted driver for inner-loop iteration; reach for the full-matrix scripts
when you need a complete host/std/arch sweep.

## Fast iteration — `ci/util/build_and_test_targets.sh`

Single wrapper around `cmake`, `ninja`, `ctest`, and `lit` for one preset at a time — the same driver
`cccl-build` uses, with `--ctest-targets` and `--lit-tests` / `--lit-precompile-tests` added for the test
runners:

- `ctest` — runs the regexes in `--ctest-targets` against the built CTest registry (CUB, Thrust, cudax, C Parallel)
- `lit` — runs `--lit-tests` (and pre-compiles `--lit-precompile-tests`) for libcudacxx

Run from the repo root, inside the devcontainer. GPU required for execution.

**CTest targets (CUB, Thrust, cudax, C Parallel):**

```
ci/util/build_and_test_targets.sh \
  --preset <name> \
  --build-targets "<target>" \
  --ctest-targets "<target>"
```

`--ctest-targets` takes a space-separated list of CTest `-R` regexes. Omit `--build-targets` if already built.

**lit targets (libcudacxx):**

```
ci/util/build_and_test_targets.sh \
  --preset libcudacxx \
  --lit-precompile-tests "std/algorithms/alg.nonmodifying/alg.any_of/any_of.pass.cpp" \
  --lit-tests           "std/algorithms/alg.nonmodifying/alg.any_of/any_of.pass.cpp"
```

Paths are relative to `libcudacxx/test/libcudacxx/`. Avoid `--build-targets "libcudacxx.cpp20.precompile.lit"` — it precompiles the entire suite.

Common preset/target pairs:

| Project    | Preset(s)                      | Target example                   |
|------------|--------------------------------|----------------------------------|
| CUB        | `cub-cpp17`, `cub-cpp20`       | `cub.cpp20.test.iterator`        |
| Thrust     | `thrust-cpp17`, `thrust-cpp20` | `thrust.cpp20.test.reduce`       |
| cudax      | `cudax`                        | `cudax.cpp20.test.async_buffer`  |
| C Parallel | `cccl-c-parallel`              | `cccl.c.test.reduce`             |

Also available: `--custom-test-cmd "<cmd>"` for an arbitrary command after CTest.

## Full matrix — `ci/test_*.sh`

Per-project scripts that test across a full host/std/arch sweep. GPU required.

```
./ci/test_<project>.sh  -cxx <compiler>  -std <std>  -arch "<arch-list>"
```

| Project    | Script            | Stds    |
|------------|-------------------|---------|
| CUB        | `test_cub`        | 17, 20  |
| Thrust     | `test_thrust`     | 17, 20  |
| libcudacxx | `test_libcudacxx` | 17, 20  |
| cudax      | `test_cudax`      | 20 only |

Test scripts build implicitly if the tree is missing. CTest preset form also works:
`ctest --preset=cub-cpp17`.

Compute-sanitizer variants: append `-compute-sanitizer-{memcheck,racecheck,initcheck,synccheck}`.
Not all projects support all tools — check `--help`.

Full test runs: 30+ min. Never cancel mid-run.

For architecture flag syntax, see `cccl-build` → `references/arch-flag.md`.

## Additional resources

- `references/docs.md` — index of CCCL test documentation.
- `references/tools.md` — all test scripts with purpose and cross-references.
- `cccl-build` → `references/build_and_test_targets_usage.md` — full `build_and_test_targets.sh` interface (shared with build).
- See `cccl-build` for building before you test.
