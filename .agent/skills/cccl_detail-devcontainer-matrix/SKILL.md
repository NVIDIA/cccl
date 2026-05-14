---
description: |
  CCCL devcontainer matrix expansion — `make_devcontainers.sh`, 60+ generated
  configs from `ci/matrix.yaml`'s `devcontainers:` section, the verify-devcontainers
  CI workflow, and `[skip-vdc]` policy. Reference for understanding the generation
  pipeline, naming convention, and when regeneration is required.
  Triggers: "make_devcontainers.sh", "regenerate devcontainers", "skip-vdc",
  "devcontainer matrix expansion", "verify-devcontainers".
---

The generated `.devcontainer/` subdirs are produced from `ci/matrix.yaml` and are not
hand-edited. This skill covers what drives them, what the CI check enforces, and when
`[skip-vdc]` is safe to apply.

## Generation pipeline

`ci/matrix.yaml` → `make_devcontainers.sh` → `.devcontainer/cuda<CTK>-<compiler>/devcontainer.json`

The `devcontainers:` section of `ci/matrix.yaml` lists `dc` and `dc_ext` job entries that
enumerate the CTK × host-compiler combinations requiring devcontainers. These entries have
no corresponding build/test scripts — they exist solely to drive container generation.

`make_devcontainers.sh` calls `.github/actions/workflow-build/build-workflow.py` with
`--devcontainer-info` to resolve aliases (`12.X`, `13.X`) and emit a JSON combination
list, then renders a `devcontainer.json` for each entry.

## Naming convention

`.devcontainer/cuda<CTK>[-ext]-<compiler><version>/`

| Suffix     | When                                                    |
|------------|-------------------------------------------------------|
| _(none)_  | Standard CTK image                                      |
| `ext`     | Extended CTK image (`dc_ext` job type; extra CUDA libraries) |

Examples: `cuda13.2-gcc14/`, `cuda12.9ext-llvm20/`, `cuda13.0-nvhpc25.11/`.

The four `cuda99.8` and `cuda99.9` entries are internal NVIDIA images (compiler versions
pulled from `cuda99_gcc_version` / `cuda99_clang_version` in `matrix.yaml`). They are
generated but excluded from the `verify-devcontainers` test matrix.

## Template

`.devcontainer/devcontainer.json` is both the base template and the default container
(updated in-place to the newest GCC × highest-CTK combination on each regeneration run).
The per-combination `devcontainer.json` files inherit its structure; `make_devcontainers.sh`
stamps each with the correct image, name, and `CCCL_*` env vars via `jq`.

Direct edits to per-combination `devcontainer.json` files are overwritten on the next
regeneration run. Edit the base template or `ci/matrix.yaml`, then regenerate.

## verify-devcontainers workflow

`.github/workflows/verify-devcontainers.yml` runs `make_devcontainers.sh --verbose --clean`
and asserts the working tree is clean. A dirty tree means the committed files are out of
sync with what the matrix produces. It then fans out across all non-`cuda99` containers and
runs `.devcontainer/verify_devcontainer.sh` inside each one.

The workflow fires when `.devcontainer/`, `ci/matrix.yaml`, or
`.github/actions/workflow-build/build-workflow.py` are modified in a PR.

## `[skip-vdc]` policy

`[skip-vdc]` in the last commit message disables the verify-devcontainers jobs for that PR run.

Safe to apply when: the PR touches neither `.devcontainer/`, `ci/matrix.yaml`, nor
`build-workflow.py`, and the devcontainer state is known-good.

Not safe to apply on PRs that modify any of those three paths — a dirty-tree failure
will surface on merge and block the branch.

## Additional resources

- `cccl-devcontainer` — step-by-step regen walkthrough and validation (its `references/regenerate.md`).
- `references/tools.md` — `make_devcontainers.sh` ownership and usage cross-reference.
