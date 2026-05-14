---
description: |
  CCCL C++ build paths — fast iteration first, full matrix when needed.
  Covers `ci/util/build_and_test_targets.sh` for single-preset targeted builds
  and `ci/build_*.sh` for full host/std/arch matrix builds.
  Triggers: "build just X", "targeted build", "build cub", "build thrust",
  "full matrix build", "compile cudax".
---

# cccl-build

Two build paths. Prefer the targeted build_and_test_targets.sh for inner-loop iteration;
reach for the `ci/{build|test}_*` scripts when you need a complete host/std/arch sweep.

## Fast iteration — `ci/util/build_and_test_targets.sh`

Single wrapper around `cmake`, `ninja`, `ctest`, and `lit` for one preset at a time:

- `cmake` — configure the preset (always runs unless cached)
- `ninja` — build the named `--build-targets`
- `ctest` — run the named `--ctest-targets` (regex list)
- `lit` — run the named `--lit-tests` / `--lit-precompile-tests` (libcudacxx)

This skill covers the configure/build flags. See `cccl-test` for the `ctest` and `lit` runners.

Run from the repo root, inside the devcontainer (or anywhere the preset's compiler is available).

```
ci/util/build_and_test_targets.sh \
  --preset <name> \
  --build-targets "<target>"
```

Common preset/target pairs:

| Project    | Preset(s)                      | Target example                  |
|------------|--------------------------------|---------------------------------|
| CUB        | `cub-cpp17`, `cub-cpp20`       | `cub.cpp20.test.iterator`       |
| Thrust     | `thrust-cpp17`, `thrust-cpp20` | `thrust.cpp20.test.reduce`      |
| cudax      | `cudax`                        | `cudax.cpp20.test.async_buffer` |
| C Parallel | `cccl-c-parallel`              | `cccl.c.test.reduce`            |
| libcudacxx | `libcudacxx`                   | use `--lit-precompile-tests`    |

Other useful flags: `--cmake-options`, `--configure-override`. Omit `--build-targets` → configure only.
Build dir: `build/${CCCL_BUILD_INFIX}/${PRESET}/`.

Wrap in the devcontainer:

```
.devcontainer/launch.sh -d --cuda 13.2 --host gcc14 -- \
  ./ci/util/build_and_test_targets.sh \
    --preset cub-cpp20 \
    --build-targets "cub.cpp20.test.iterator"
```

## Full matrix — `ci/build_*.sh`

Per-project scripts that build across a full host/std/arch sweep. No GPU required for build.

```
./ci/build_<project>.sh  [-cxx <compiler>] [-std <std>] [-arch "<arch-list>"]
```

| Project    | Script                  | Stds    |
|------------|-------------------------|---------|
| CUB        | `build_cub`             | 17, 20  |
| Thrust     | `build_thrust`          | 17, 20  |
| libcudacxx | `build_libcudacxx`      | 17, 20  |
| cudax      | `build_cudax`           | 20 only |
| C Parallel | `build_cccl_c_parallel` | 17 only |

Architecture flag (`-arch`): semicolon-separated CMake `CUDA_ARCHITECTURES` list.
`native` or `"80"` is much faster than `all-major-cccl`. See `references/arch-flag.md` for syntax forms.

Full builds: 60+ min. Never cancel mid-run.
`sccache` is enabled in the devcontainer (CCCL-team bucket auth).

## Additional resources

- `references/arch-flag.md` — architecture flag forms (`<XX>`, `<XX-real>`, `<XX-virtual>`, `native`, `all-major-cccl`)
- `references/docs.md` — index of CCCL build documentation.
- `references/tools.md` — all build scripts with purpose and ownership.
- `references/build_and_test_targets_usage.md` — `ci/util/build_and_test_targets.sh` interface and examples.
- `references/build_common.sh_usage.md` — `ci/build_common.sh` options, env vars, and helper functions.
- See `cccl-test` for running tests after a build.
