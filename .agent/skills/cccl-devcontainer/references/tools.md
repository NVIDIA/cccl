# Tool index — cccl-devcontainer

## Owned (canonical reference lives here)

| Tool | Purpose | Detail |
|------|---------|--------|
| `.devcontainer/launch.sh` | Launch a devcontainer (Docker or VSCode) for a given CTK version + host compiler combination. The primary entry point for all devcontainer-based builds and tests. | `references/launch_usage.md` |
| `.devcontainer/docker-entrypoint.sh` | Docker container startup hook: sets environment, sources sccache config, runs requested command. Invoked by Docker, not directly. | sourced by Docker; not user-invoked |
| `.devcontainer/cccl-entrypoint.sh` | CCCL-specific container init: sets `CCCL_BUILD_INFIX`, sources build environment. Sourced inside the container. | sourced by container; not user-invoked |
| `.devcontainer/verify_devcontainer.sh` | Verifies a named devcontainer config is well-formed and builds successfully. Used by the `verify-devcontainers` CI workflow. | CI-internal |

## Used (canonical reference lives in another skill)

| Tool | Purpose | Reference |
|------|---------|-----------|
| `.devcontainer/make_devcontainers.sh` | Generates 60+ devcontainer subdirectory configs from `ci/matrix.yaml`. | `cccl_detail-devcontainer-matrix` → `references/tools.md` |
