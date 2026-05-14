---
description: "Run a git bisect on CCCL to find the commit that introduced a regression. Cloud route: dispatch `.github/workflows/git-bisect.yml` on a GPU runner. Local route: invoke `ci/util/git_bisect.sh` via `.devcontainer/launch.sh`. Walks through preset, build/test targets, good/bad refs. Triggers: \"bisect this regression\", \"find when X broke\", \"git bisect\"."
---

# cccl-bisect

Bisects are slow. Restrict build and test targets to the smallest set that reliably reproduces the regression.

## Sources of truth

- `.github/workflows/git-bisect.yml`
- `ci/util/git_bisect.sh`
- `ci/util/build_and_test_targets.sh`
- `docs/cccl/development/build_and_bisect_tools.rst`

## Inputs needed

- **`preset`** — CMake preset (e.g. `cub-cpp20`, `thrust-cpp17`, `libcudacxx`, `cudax`). `cmake --list-presets` enumerates them.
- **`build_targets`** — space-separated ninja targets.
- **`ctest_targets`** — space-separated CTest `-R` regexes. Optional.
- **`lit_precompile_tests` / `lit_tests`** — space-separated libcudacxx lit paths relative to `libcudacxx/test/libcudacxx/`. Optional.
- **`good_ref`** / **`bad_ref`** — commit/tag/branch, or `-Nd` ("N days ago on main", e.g. `-7d`), or empty (defaults: latest release tag / `main`).
- **`cmake_options`** — extra `-D…=…` flags. Optional.
- **`launch_args`** — extra `--cuda X` / `--host Y` for devcontainer. Optional.

Route ambiguous inputs through `cccl-clarify`.

## Route 1 — cloud dispatch

```
gh workflow run git-bisect.yml --repo NVIDIA/cccl --ref <branch> \
  -f runner='<runner-label>' \
  -f preset='<preset>' \
  -f build_targets='<targets>' \
  -f ctest_targets='<regex>' \
  -f good_ref='<good>' \
  -f bad_ref='<bad>'
```

Runner labels:

- `linux-amd64-cpu16` — 16-core CPU box (build-only bisects).
- `linux-amd64-gpu-rtxa6000-latest-1` — RTX A6000, 1 GPU (test bisects).
- Others: see the workflow file inputs.

Return the run URL.

## Route 2 — local

Requires Docker.

```
.devcontainer/launch.sh -d <launch_args> --gpus all \
  -- ./ci/util/git_bisect.sh \
    --summary-file /tmp/shared/summary.md \
    --good-ref '<good>' \
    --bad-ref '<bad>' \
    --preset '<preset>' \
    --build-targets '<targets>' \
    --ctest-targets '<regex>'
```

## Output

Both routes write a `summary.md` capturing the found-bad commit (hash, author, message), the build/test command that distinguishes good from bad, and the bisect log. Cloud route surfaces a "Bisection Results" URL in the GHA step summary.

## Additional resources

- `references/docs.md` — index of CCCL bisect documentation.
- `references/tools.md` — `git_bisect.sh` and cross-referenced build tools.
- `references/git_bisect_usage.md` — `ci/util/git_bisect.sh` interface and examples.
- `cccl-build` → `references/build_and_test_targets_usage.md` — build/test flags shared with bisect.
