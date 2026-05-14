# `ci/util/build_and_test_targets.sh` usage

Unified driver for configure, build, and test in one pass. The inner-loop tool for targeted builds
and test runs against a single CMake preset. Wraps `cmake`, `ninja`, `ctest`, and `lit` in sequence;
stops and reports on the first failure with an elapsed-time banner.

## Location

`ci/util/build_and_test_targets.sh`. Run from the repo root, inside the devcontainer (or any
environment where the preset's compilers are on `PATH`). No GPU required for build-only invocations;
GPU required for `--ctest-targets` and `--lit-tests`.

## Interface

```
Usage: ./ci/util/build_and_test_targets.sh [--preset NAME | --configure-override CMD] [options]

Options:
  -h, --help                Show this help and exit
  --preset NAME             CMake preset
  --cmake-options STR       Extra options passed to CMake preset configure (optional)
  --configure-override CMD  Command to run for configuration instead of cmake preset
                            If set, --preset and --cmake-options will be ignored
  --build-targets STR       Space separated ninja build targets (optional)
                            If omitted, no targets will be built -- explicitly specify 'all' if needed.
  --ctest-targets STR       Space separated CTest -R regex patterns (optional)
                            If omitted, no tests will be run -- explicitly specify '.' to run all.
  --lit-precompile-tests STR  Space-separated libcudacxx lit test paths to precompile without execution (optional)
                              e.g. 'cuda/utility/basic_any.pass.cpp'
  --lit-tests STR            Space-separated libcudacxx lit test paths to execute (optional)
                              e.g. 'cuda/utility/basic_any.pass.cpp'
  --custom-test-cmd CMD     Custom command run after build and tests (optional)
```

## Options

| Flag                    | Required? | Description                                                              |
|-------------------------|-----------|--------------------------------------------------------------------------|
| `--preset`              | Yes*      | CMake preset name. Mutually exclusive with `--configure-override`.       |
| `--configure-override`  | Yes*      | Shell command replacing the `cmake --preset` step. Ignores `--preset`.   |
| `--cmake-options`       | No        | Extra `-D…` flags appended to `cmake --preset`. Space-separated string.  |
| `--build-targets`       | No        | Ninja targets to build. Space-separated; quoted. Omit = configure only.  |
| `--ctest-targets`       | No        | CTest `-R` regex patterns. Space-separated; each runs a separate `ctest`.|
| `--lit-precompile-tests`| No        | Lit paths to precompile (no execution). Relative to `libcudacxx/test/libcudacxx/`. |
| `--lit-tests`           | No        | Lit paths to execute. Relative to `libcudacxx/test/libcudacxx/`.         |
| `--custom-test-cmd`     | No        | Arbitrary command run after all other test steps.                        |

\* One of `--preset` or `--configure-override` is required.

## Environment

| Variable            | Default               | Effect                                                          |
|---------------------|-----------------------|-----------------------------------------------------------------|
| `CCCL_BUILD_INFIX`  | `""`                  | Subdirectory under `build/` for this devcontainer's artifacts.  |

Build output lands in `build/${CCCL_BUILD_INFIX}/${PRESET}/`. The `build/latest` symlink always points
to the most recent `build/${CCCL_BUILD_INFIX}/` directory; `build/preset-latest` to the most recent
preset subdirectory within it.

## Examples

```bash
# Build a single CUB target
ci/util/build_and_test_targets.sh \
  --preset cub-cpp20 \
  --build-targets "cub.cpp20.test.iterator"

# Build then run a CTest target
ci/util/build_and_test_targets.sh \
  --preset cub-cpp20 \
  --build-targets "cub.cpp20.test.iterator" \
  --ctest-targets "cub.cpp20.test.iterator"

# libcudacxx lit: precompile then execute one test
ci/util/build_and_test_targets.sh \
  --preset libcudacxx \
  --lit-precompile-tests "std/algorithms/alg.nonmodifying/alg.any_of/any_of.pass.cpp" \
  --lit-tests           "std/algorithms/alg.nonmodifying/alg.any_of/any_of.pass.cpp"

# Wrapped in the devcontainer
.devcontainer/launch.sh -d --cuda 13.2 --host gcc14 -- \
  ./ci/util/build_and_test_targets.sh \
    --preset thrust-cpp20 \
    --build-targets "thrust.cpp20.test.reduce"
```

## Wraps / calls

- `cmake --preset` — configure step (or `--configure-override` replacement)
- `ninja -C <build_dir>` — build step for each `--build-targets` entry
- `ctest --test-dir <build_dir> -R <pattern>` — one invocation per `--ctest-targets` entry
- `lit -v` — one invocation per `--lit-tests` entry; precompile pass uses `-Dexecutor=NoopExecutor()`

## Notes / gotchas

- `--ctest-targets` runs one `ctest -R <pattern>` per entry; patterns are regex, not glob.
- `--lit-tests` paths are relative to `libcudacxx/test/libcudacxx/`. Absolute paths will fail.
- Avoid `--build-targets "libcudacxx.cpp20.precompile.lit"` — it precompiles the entire lit suite.
- `LIBCUDACXX_SITE_CONFIG` is set automatically from the build directory; do not override it.
- The script exits on the first failure with a colored banner and elapsed time; subsequent steps are skipped.
