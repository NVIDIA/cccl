# Release cut checklist

Three GitHub Actions workflows orchestrate CCCL releases. Run them in order.

## Phase 0 — Version bump on main (pre-release)

Trigger `.github/workflows/update-branch-version.yml` (workflow: "Release: 0. Update version
in target branch") against `main` with the *next* version (e.g., `3.5.0`). This runs
`ci/update_version.sh` and opens a PR.

Files updated by `ci/update_version.sh`:
| File                                       | Field           |
|--------------------------------------------|-----------------|
| `cccl-version.json`                        | `full`, `major`, `minor`, `patch` |
| `libcudacxx/include/cuda/std/__cccl/version.h` | `CCCL_VERSION`  |
| `thrust/thrust/version.h`                  | `THRUST_VERSION` |
| `cub/cub/version.cuh`                      | `CUB_VERSION`   |
| `lib/cmake/cccl/cccl-config-version.cmake` | version vars    |
| `lib/cmake/cub/cub-config-version.cmake`   | version vars    |
| `lib/cmake/libcudacxx/libcudacxx-config-version.cmake` | version vars |
| `lib/cmake/thrust/thrust-config-version.cmake` | version vars |
| `lib/cmake/cudax/cudax-config-version.cmake` | version vars |
| `python/cuda_cccl/cuda/cccl/_version.py`   | `__version__`   |
| `docs/VERSION.md`                          | major.minor     |

Merge the PR from this workflow before proceeding.

## Phase 1 — Begin release cycle

Trigger `.github/workflows/release-create-new.yml` (workflow: "Release: 1. Begin Release
Cycle") from the commit on `main` that should be the release base. Provide `main_version`
(the next version for `main` after this release, e.g., `3.6.0`).

This workflow:
1. Creates `branch/3.5.x` from the selected ref.
2. Updates `main` to `main_version`.
3. Opens PRs for both operations.

## Phase 2 — RC updates (optional)

For release candidates, trigger `.github/workflows/release-update-rc.yml` on the release
branch. Provide the RC version string (e.g., `3.5.0rc1`). Opens a PR against the release
branch.

## Phase 3 — Finalize

Trigger `.github/workflows/release-finalize.yml` on the release branch. This workflow
publishes the final tag and release artifacts.

## Phase 4 — Wheel builds

Trigger `.github/workflows/release-wheels.yml` after finalization to build and publish
Python wheel artifacts.

## Key invariants

- Always verify `cccl-version.json` is correct before triggering Phase 1 — the workflow
  reads it as the current version.
- The `update-branch-version.yml` workflow is the only safe way to bump version numbers;
  never edit version files by hand across the 10+ locations they appear.
- Release branches follow the pattern `branch/{major}.{minor}.x`.
