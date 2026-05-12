---
name: cccl-cpp-builds
description: "Build and test CCCL's C++ libraries (libcudacxx, CUB, Thrust, cudax, C Parallel) — per-project `ci/build_*.sh` and `ci/test_*.sh` full-matrix scripts, architecture conventions, and pointers to the targeted-build alternative. Use when the user wants to build or test a CCCL C++ library across a full host/std/arch matrix, or asks about architecture flag syntax. Trigger phrases: \"build cub\", \"test libcudacxx\", \"build thrust\", \"full matrix build\", \"compile cudax\", \"cuda architectures\". For SINGLE-target fast iteration use `cccl-build-and-test-targets` instead."
---

# cccl-cpp-builds

Per-project full-matrix build + test scripts under `ci/`. Flags: host compiler, C++ standard, GPU architectures.

Full builds: 60+ min build, 30+ min test — never cancel. For single targets, use `cccl-build-and-test-targets`.

## Scripts

```
./ci/build_<project>.sh  [-cxx <compiler>] [-std <std>] [-arch "<arch-list>"]   # no GPU
./ci/test_<project>.sh    -cxx <compiler>   -std <std>   -arch "<arch-list>"    # GPU required
```

| Project           | Build / test scripts        | Stds      |
|-------------------|-----------------------------|-----------|
| CUB               | `build_cub`, `test_cub`     | 17, 20    |
| Thrust            | `build_thrust`, `test_thrust` | 17, 20  |
| libcudacxx        | `build_libcudacxx`, `test_libcudacxx` | 17, 20 |
| cudax             | `build_cudax`, `test_cudax` | 20 only   |
| C Parallel        | `build_cccl_c_parallel`     | 17 only   |

Test scripts build implicitly if the tree is missing. CTest preset form (e.g. `ctest --preset=cub-cpp17`) also
works.

Compute-sanitizer variants: append `-compute-sanitizer-{memcheck,racecheck,initcheck,synccheck}`. Not all
projects support all tools — check `--help`.

## Flags

- **`-cxx`** — host compiler (`g++`, `clang++`, `msvc14.39`).
- **`-std`** — C++ standard (`17` or `20`, subject to project limits above).
- **`-arch`** — semicolon-separated CUDA architecture list (CMake `CUDA_ARCHITECTURES`):

  | Form             | Generates             |
  |------------------|-----------------------|
  | `<XX>`           | PTX + SASS for SM XX  |
  | `<XX-real>`      | SASS only             |
  | `<XX-virtual>`   | PTX only              |
  | `native`         | Detect host GPU       |
  | `all-major-cccl` | Default for PR builds |

  Examples: `"native"`, `"80"`, `"70;75;80-virtual"`.

## Performance

- `sccache` is enabled in the devcontainer (CCCL-team bucket auth).
- Limit `-arch` — `"native"` or `"80"` is much faster than `"all-major-cccl"`.
- Build scripts already parallelize via ninja.
