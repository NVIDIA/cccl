# Tool index — cccl_detail-devcontainer-matrix

## Owned (canonical reference lives here)

| Tool | Purpose | Detail |
|------|---------|--------|
| `.devcontainer/make_devcontainers.sh` | Generates all `.devcontainer/{cuda-X.Y}-{compiler}/` subdirectories from the `devcontainers:` section of `ci/matrix.yaml`. Must be re-run whenever `ci/matrix.yaml` adds/changes/removes devcontainer entries. | see `cccl-devcontainer` → `references/regenerate.md` for the step-by-step regen workflow |

## Used (canonical reference lives in another skill)

| Tool | Purpose | Reference |
|------|---------|-----------|
| `.devcontainer/launch.sh` | Used to verify generated containers build and launch correctly after regeneration. | `cccl-devcontainer` → `references/launch_usage.md` |
