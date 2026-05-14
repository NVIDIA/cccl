# `ci/util/git_bisect.sh` usage

Automates `git bisect` for CCCL regression isolation. Checks out each candidate commit, runs the
specified build and test commands (same flag set as `build_and_test_targets.sh`), and produces a
Markdown summary identifying the first bad commit, the distinguishing command, and the full bisect log.

## Location

`ci/util/git_bisect.sh`. Run from the repo root, inside the devcontainer. GPU required when
`--ctest-targets` or `--lit-tests` are specified.

## Interface

```
Usage: ./ci/util/git_bisect.sh [--preset NAME | --configure-override CMD] [options]

Generic Options:

  -h, --help             Show this help and exit

Bisection Options:

  --good-ref STR         Good ref/sha/tag/branch. Defaults to latest release tag.
                         Accepts '-Nd' (e.g., '-14d') to mean 'origin/main as of N days ago'.
  --bad-ref STR          Bad ref/sha/tag/branch. Defaults to origin/main.
                         Accepts '-Nd' (e.g., '-14d') to mean 'origin/main as of N days ago'.
  --summary-file PATH    Markdown summary output path (optional)
                         No summary file will be generated if this is omitted.

Build / Test Options:

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
  --repeat N               Re-run the build/test for passing commits N times (default: 1)
```

## Options

| Flag                    | Required? | Description                                                               |
|-------------------------|-----------|---------------------------------------------------------------------------|
| `--preset`              | Yes*      | CMake preset. Same as `build_and_test_targets.sh`.                        |
| `--configure-override`  | Yes*      | Shell command replacing `cmake --preset`. Mutually exclusive with preset. |
| `--good-ref`            | No        | Last-known-good commit/tag/branch. Defaults to latest release tag.        |
| `--bad-ref`             | No        | First-known-bad commit/tag/branch. Defaults to `origin/main`.             |
| `--summary-file`        | No        | Path for Markdown output. Omit to skip file generation.                   |
| `--build-targets`       | No        | Space-separated ninja targets (quoted string).                            |
| `--ctest-targets`       | No        | Space-separated CTest `-R` regex patterns (quoted string).                |
| `--lit-precompile-tests`| No        | Lit paths to precompile; relative to `libcudacxx/test/libcudacxx/`.       |
| `--lit-tests`           | No        | Lit paths to execute; relative to `libcudacxx/test/libcudacxx/`.          |
| `--cmake-options`       | No        | Extra `-D…` flags for `cmake --preset`.                                   |
| `--custom-test-cmd`     | No        | Arbitrary command run after build and tests.                              |
| `--repeat`              | No        | Re-run passing commits N times to guard against flakes. Default: `1`.     |

\* One of `--preset` or `--configure-override` is required.

## Examples

```bash
# Bisect a CUB test failure against the last 14 days of main
.devcontainer/launch.sh -d --cuda 13.2 --host gcc14 --gpus all -- \
  ./ci/util/git_bisect.sh \
    --summary-file /tmp/shared/bisect-summary.md \
    --good-ref '-14d' \
    --preset 'cub-cpp20' \
    --build-targets 'cub.cpp20.test.iterator' \
    --ctest-targets 'cub.cpp20.test.iterator'

# Bisect with explicit good/bad SHAs, no GPU (build-only)
.devcontainer/launch.sh -d --cuda 13.2 --host gcc14 -- \
  ./ci/util/git_bisect.sh \
    --good-ref 'v2.7.0' \
    --bad-ref 'main' \
    --preset 'cub-cpp20' \
    --build-targets 'cub.cpp20.test.iterator'

# Cloud dispatch via GitHub Actions
gh workflow run git-bisect.yml --repo NVIDIA/cccl --ref main \
  -f runner='linux-amd64-gpu-rtxa6000-latest-1' \
  -f preset='cub-cpp20' \
  -f build_targets='cub.cpp20.test.iterator' \
  -f ctest_targets='cub.cpp20.test.iterator' \
  -f good_ref='-14d' \
  -f bad_ref='main'
```

## Wraps / calls

- `ci/util/build_and_test_targets.sh` — for each candidate commit's build+test step
- `git bisect` — standard git bisect mechanics (start, good, bad, run, reset)

## Notes / gotchas

- Narrow `--build-targets` and `--ctest-targets` to the smallest set that reproduces the failure.
  Each bisect step is a full configure+build+test; broad targets multiply bisect time significantly.
- `--good-ref '-Nd'` resolves to the state of `origin/main` as of N days ago, not a local branch.
- `--repeat N` reruns the test N times on commits that pass, useful when the failure is intermittent.
- The summary file captures: bad commit hash/author/message, the distinguishing command, and the
  full `git bisect log` output. Useful for surfacing in PR comments or CI step summaries.
- Cloud route (`.github/workflows/git-bisect.yml`) produces a "Bisection Results" link in the GHA
  step summary; prefer it for long bisects to avoid local GPU contention.
