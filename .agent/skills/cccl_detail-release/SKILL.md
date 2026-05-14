---
description: |
  CCCL versioning and release pipeline — `cccl-version.json`, `ci/update_version.sh`,
  release workflows (numbered 0–3), Python wheel versioning via `setuptools_scm`,
  header version macros, branch/tag conventions, and the RC → final promotion path.
  Triggers: "how does CCCL versioning work", "how do I cut a release", "what is cccl-version.json",
  "release branch convention", "how are wheels versioned".
---

Single source of truth for CCCL version numbers is `cccl-version.json` at the repo root.
The `ci/update_version.sh` script propagates a new version to every consumer in one pass.
Releases follow a numbered four-step workflow, each a manual `workflow_dispatch`.

## Version pipeline

`cccl-version.json` holds `{ "full", "major", "minor", "patch" }`.
`ci/update_version.sh <major> <minor> <patch>` writes every downstream target atomically.
`--dry-run` shows a diff without touching the tree.

Targets updated by `update_version.sh`:

| Target                                      | Format                                        |
|---------------------------------------------|-----------------------------------------------|
| `cccl-version.json`                         | plain `x.y.z`                                 |
| `libcudacxx/include/cuda/std/__cccl/version.h` | `CCCL_VERSION` → `MMMmmmppp` (9 digits)   |
| `thrust/thrust/version.h`                   | `THRUST_VERSION` → `MMMmmmpp` (8 digits)      |
| `cub/cub/version.cuh`                       | `CUB_VERSION` → same 8-digit scheme            |
| `lib/cmake/*/`-config-version.cmake (×5)   | separate `MAJOR`/`MINOR`/`PATCH` vars         |
| `python/cuda_cccl/cuda/cccl/_version.py`   | `__version__ = "x.y.z"`                       |
| `docs/VERSION.md`                           | `x.y` (major.minor only)                      |

The script guards against downgrades: it compares the encoded new version against both
the current `CCCL_VERSION` define and the latest `git tag`.

## Python wheel versioning

`python/cuda_cccl/pyproject.toml` declares `dynamic = ["version"]` and sources it via
`scikit_build_core.metadata.setuptools_scm` with `root = "../.."`.
At build time `setuptools_scm` derives the version from the nearest git tag.
Release builds tag first (`vX.Y.Z`), then build wheels — the tag drives the wheel version.
`python/cuda_cccl/cuda/cccl/_version.py` carries a static fallback (`__version__`) updated
by `update_version.sh` for editable / non-tag installs.

## Branch and tag conventions

| Artifact                | Pattern                          | Example                        |
|-------------------------|----------------------------------|--------------------------------|
| Release branch          | `branch/{major}.{minor}.x`       | `branch/3.4.x`                 |
| Release candidate tag   | `v{full}-rc{N}`                  | `v3.4.0-rc0`                   |
| Final release tag       | `v{full}`                        | `v3.4.0`                       |
| Version-bump PR branch  | `pr/ver/{branch}-v{version}`     | `pr/ver/branch/3.4.x-v3.4.1`   |

## Release workflows (numbered steps)

All four workflows are `workflow_dispatch` only — no automatic triggers.

**Step 0 — Update version in target branch** (`update-branch-version.yml`)
Creates a `pr/ver/…` branch, runs `update_version.sh`, opens a PR into the target branch.
Shared composite action: `.github/actions/version-update/action.yml`.

**Step 1 — Begin Release Cycle** (`release-create-new.yml`)
Run on `main` (or the commit to branch from). Creates `branch/{major}.{minor}.x` from the
current SHA, then calls the version-update action to bump `main` to the next version.

**Step 2 — Test and Tag New RC** (`release-update-rc.yml`)
Run on `branch/{major}.{minor}.x`. Reads version from `cccl-version.json`, auto-increments
the RC counter (scanning existing `vX.Y.Z-rcN` tags), pushes `v{full}-rc{N}`.

**Step 3 — Create Final Release** (`release-finalize.yml`)
Run on an RC tag (`vX.Y.Z-rcN`). Validates `cccl-version.json` matches the tag, generates
source + install archives via `ci/install_cccl.sh`, pushes `vX.Y.Z` tag from the RC tag,
creates a draft GitHub release. Optional `create_patch_version` input opens a patch-bump PR.

**Wheel publish** (`release-wheels.yml`)
Separate manual workflow. Takes a prior GHA `run-id` (the validated build artifacts) and a
destination (`pypi` or `testpypi`). Publishes via `pypa/gh-action-pypi-publish`.

## Hard prohibitions

- Never run `update_version.sh` with a version lower than the current `CCCL_VERSION` or the
  latest git tag — the script rejects it, but don't try to work around the guard.
- Step 3 must be started on an RC tag, not a branch ref.
- Wheel publish must reference a previously validated build run — don't build and publish
  in the same step.

## Additional resources

- `references/docs.md` — index of release workflow and versioning documentation.
