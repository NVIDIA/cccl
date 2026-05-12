---
name: cccl-devcontainers
description: "Use CCCL's `.devcontainer/launch.sh` to run one-off bash sessions, builds, or tests inside a CCCL-configured container with a chosen CUDA toolkit and host compiler. Covers the `-d` / `--cuda` / `--host` / `--gpus` / `--env` / `--volume` argument conventions and the `CCCL_BUILD_INFIX` already-in-container check. Use when the user wants to build/test in a clean, reproducible environment, run a quick experiment with a specific toolchain, or escape from host environment problems. Trigger phrases: \"run in devcontainer\", \"launch the container\", \"build with cuda 13.2\", \"open a shell with gcc 14\"."
---

# cccl-devcontainers

`.devcontainer/launch.sh` boots a Docker container preconfigured with a chosen CUDA toolkit and host compiler,
mounts the repo, and either drops into a shell or runs a script. **Linux-only** — Linux host, Linux container.
Windows / MSVC builds run outside the devcontainer.

## Flags

| Flag                     | Purpose                                  |
|--------------------------|------------------------------------------|
| `-d`, `--docker`         | Run without VSCode (required for agents) |
| `--cuda <version>`       | CUDA toolkit (e.g. `13.2`, `12.9`)       |
| `--cuda-ext`             | Image with extended CTK libraries        |
| `--host <compiler>`      | Host compiler (`gcc14`, `clang17`)       |
| `--gpus <request>`       | GPU passthrough (`all` for everything)   |
| `-e`, `--env KEY=VAL`    | Inject env var                           |
| `-v`, `--volume SRC:DST` | Mount additional path                    |
| `-- <script> [args]`     | Run script inside container after setup  |

Examples:

```
.devcontainer/launch.sh -d --cuda 13.2 --host gcc14
.devcontainer/launch.sh -d --cuda 12.9 --host gcc13 -- ./ci/build_cub.sh -cxx g++ -std 17 -arch native
.devcontainer/launch.sh -d --gpus all -- ./ci/util/build_and_test_targets.sh --preset cub-cpp20 --build-targets "cub.cpp20.test.iterator"
```

## Already inside a container?

`CCCL_BUILD_INFIX` is set inside the container. Before launching:

```
echo "$CCCL_BUILD_INFIX"
```

Non-empty → already inside; run the command directly. Nested launches don't work.

First launch pulls the image; subsequent launches use cache.

## Updating devcontainers

Per-combination subdirs (`.devcontainer/cuda<version>-<host>/`) and their `devcontainer.json` files are
**generated** — direct edits get overwritten. To change the set of available containers:

1. Edit `ci/matrix.yaml` — the `dc` (and `dc_ext` for extended-CTK) entries control which CUDA × host-compiler
   combinations exist.
2. If the template itself needs changing, edit the base `.devcontainer/devcontainer.json`.
3. Run `.devcontainer/make_devcontainers.sh --clean` from the repo root to regenerate per-combination subdirs and
   prune stale ones.
4. Push; CI's "Validate Devcontainer" jobs run.

`[skip-vdc]` blocks Validate Devcontainer jobs. Don't use it on PRs that modify `.devcontainer/`, `ci/`, or
`.github/`.
