---
description: |
  Cross-functional infrastructure maintenance — tasks that fan out across CI, devcontainers,
  CMake, and release tooling simultaneously. Covers CTK bumps, host-compiler additions,
  release cuts, and project add/remove.
  Triggers: "bump CTK", "add compiler version", "cut a release", "add a new project",
  "infra overview", "what touches the matrix".
---

Entry point for cross-cutting infrastructure work. Individual subsystem questions go to
the per-subsystem skills listed below; this skill handles tasks that touch more than one
at once and provides the ordered playbooks for each.

## Subsystem map

| Subsystem                      | Skill                              | Canonical file(s)                                                                     |
|--------------------------------|------------------------------------|---------------------------------------------------------------------------------------|
| CI matrix / job dispatch       | `cccl-ci`                          | `ci/matrix.yaml`, `.github/actions/workflow-build/build-workflow.py`                   |
| Devcontainer generation        | `cccl-devcontainer`                | `.devcontainer/make_devcontainers.sh`, `devcontainer.json`                            |
| CMake presets / options        | `cccl-cmake`                       | `CMakePresets.json`                                                                   |
| Pre-commit linters             | `cccl-precommit`                   | `.pre-commit-config.yaml`                                                             |
| Release workflows              | `cccl_detail-release`              | `ci/update_version.sh`, `cccl-version.json`, `.github/workflows/release-*.yml`         |
| GitHub workflow templates      | `cccl_detail-github`               | `.github/workflows/`, `.github/actions/`                                              |
| Examples / packaging           | `cccl_detail-examples`             | `examples/`, `test/cmake/`, `ci/test/`                                                |
| Devcontainer matrix deep-dive  | `cccl_detail-devcontainer-matrix`  | `.devcontainer/make_devcontainers.sh`, `ctk_versions` / `host_compilers` in `ci/matrix.yaml` |

## Task fanout

| Task              | Files touched                                                                                                                                                       | Playbook                       |
|-------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------|
| Add a CTK version | `ci/matrix.yaml` (`ctk_versions`, `devcontainer_version`, workflow rows), `.devcontainer/` (regen), `.github/workflows/verify-devcontainers.yml`                  | `references/ctk-bump.md`       |
| Add/remove a host compiler | `ci/matrix.yaml` (`host_compilers`, `cuda99_*_version`, workflow rows), `.devcontainer/` (regen)                                                           | `references/compiler-bump.md`  |
| Cut a release     | `cccl-version.json`, `ci/update_version.sh` targets, `lib/cmake/*/`, per-library `version.h` / `.cuh`, `docs/VERSION.md`, `python/cuda_cccl/cuda/cccl/_version.py`, `.github/workflows/release-*.yml` | `references/release-cut.md`    |
| Add a project     | `ci/matrix.yaml` (workflow rows, `jobs:` section), `ci/project_files_and_dependencies.yaml` (new project key + dependency chain), `CMakePresets.json` (new preset), `ci/build_<proj>.sh` + `ci/test_<proj>.sh` | `references/project-add.md`    |
| Remove a project  | Inverse of add: remove matrix rows, yaml entry, presets, build/test scripts, update dependents                                                                    | `references/project-add.md`    |

## Key matrix.yaml facts

`ctk_versions` maps real CTK versions to aliases. `12.X` and `13.X` are aliases for the newest
patch release in each major line — update the alias target when adding a new patch. `devcontainer_version`
at the top of `matrix.yaml` pins the rapidsai/devcontainers image tag; bump it alongside new
CTK/compiler additions.

After any edit to `ctk_versions` or `host_compilers`, regenerate devcontainers:

```
cd .devcontainer
bash make_devcontainers.sh --clean
```

## Hard prohibitions

- Never merge a PR while `workflows.override` is non-empty in `ci/matrix.yaml`.
- Never merge a PR with `[skip-*]` tags in the last commit message.
- Never edit individual `.devcontainer/<name>/devcontainer.json` files by hand — always regenerate via `make_devcontainers.sh`.
- Never bump `cccl-version.json` by hand — use `ci/update_version.sh` or the `update-branch-version.yml` workflow.

## Additional resources

- `references/ctk-bump.md` — ordered checklist for adding a new CTK version
- `references/compiler-bump.md` — ordered checklist for adding a new host compiler
- `references/release-cut.md` — release cycle steps (branch cut through finalization)
- `references/project-add.md` — adding or removing a CCCL project
- `references/docs.md` — index of maintainer/infra documentation.
- `references/tools.md` — infra scripts (version management, utilities, downstream testing).
