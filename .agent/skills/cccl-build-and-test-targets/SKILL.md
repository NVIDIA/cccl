---
name: cccl-build-and-test-targets
description: "Reference for `ci/util/build_and_test_targets.sh` — CCCL's preset-driven configure/build/test driver used by CI, the bisect workflow, and ad-hoc local runs. Covers `--preset`, `--cmake-options`, `--configure-override`, `--build-targets`, `--ctest-targets`, `--lit-precompile-tests`, `--lit-tests`, `--custom-test-cmd`. Use when the user wants to build or test a specific target without running the full CI matrix. Trigger phrases: \"build just X\", \"run test Y\", \"targeted build\", \"how do I run the cub tests\"."
---

# cccl-build-and-test-targets

`ci/util/build_and_test_targets.sh` configures, builds, and tests a CMake preset with the targets you specify.
Run it from the repo root, inside the devcontainer (or anywhere the preset's compiler is available).

## Flags

| Flag                               | Effect                                                                                          |
|------------------------------------|-------------------------------------------------------------------------------------------------|
| `--preset <name>`                  | CMake preset (or use `--configure-override` instead)                                            |
| `--cmake-options "<flags>"`        | Extra `-D…=…` flags appended to preset configure                                                |
| `--configure-override "<cmd>"`     | Custom configure command (overrides `--preset` and `--cmake-options`)                           |
| `--build-targets "<targets>"`      | Space-separated ninja targets. Omit to skip build (`"all"` for everything)                      |
| `--ctest-targets "<regex>"`        | Space-separated CTest `-R` regexes. Omit to skip tests (`"."` for all)                          |
| `--lit-precompile-tests "<paths>"` | libcudacxx lit paths to compile without execution (relative to `libcudacxx/test/libcudacxx/`)   |
| `--lit-tests "<paths>"`            | libcudacxx lit paths to compile AND execute                                                     |
| `--custom-test-cmd "<cmd>"`        | Arbitrary command after tests                                                                   |

`--build-targets` and `--ctest-targets` are opt-in. Omit → nothing builds or tests; the script just configures.

## Common patterns

Most cases: pick the preset and pass the target as both `--build-targets` and `--ctest-targets`:

```
ci/util/build_and_test_targets.sh \
  --preset <preset> \
  --build-targets "<target>" \
  --ctest-targets "<target>"
```

| Project    | Preset(s)                        | Target example                |
|------------|----------------------------------|-------------------------------|
| CUB        | `cub-cpp17`, `cub-cpp20`         | `cub.cpp20.test.iterator`     |
| Thrust     | `thrust-cpp17`, `thrust-cpp20`   | `thrust.cpp20.test.reduce`    |
| cudax      | `cudax`                          | `cudax.cpp20.test.async_buffer` |
| C Parallel | `cccl-c-parallel`                | `cccl.c.test.reduce`          |

libcudacxx is lit-driven — use `--lit-precompile-tests` and `--lit-tests` instead of `--build-targets`:

```
ci/util/build_and_test_targets.sh \
  --preset libcudacxx \
  --lit-precompile-tests "std/algorithms/alg.nonmodifying/alg.any_of/any_of.pass.cpp" \
  --lit-tests           "std/algorithms/alg.nonmodifying/alg.any_of/any_of.pass.cpp"
```

Avoid `--build-targets "libcudacxx.cpp20.precompile.lit"` — it precompiles the entire test suite.

## Output

Build dir at `build/${CCCL_BUILD_INFIX}/${PRESET}/` (parsed from the cmake configure log line
`-- Build files have been written to:`). Phase-by-phase elapsed time printed with emoji status markers.

## Wrapping in the devcontainer

```
.devcontainer/launch.sh -d --cuda 13.2 --host gcc14 -- \
  ./ci/util/build_and_test_targets.sh \
    --preset cub-cpp20 \
    --build-targets "cub.cpp20.test.iterator"
```

## vs full-matrix scripts

- `build_and_test_targets.sh` — single preset, named targets. Fast iteration.
- `./ci/build_<project>.sh` / `./ci/test_<project>.sh` — full build/test cycles across host/std/arch matrix. Slow.
  See `cccl-cpp-builds`.
