---
description: |
  CCCL's `.devcontainer/launch.sh` — launch a Docker container with a chosen CUDA toolkit
  and host compiler, mount the repo, and run a shell or script. Linux-only. Covers flag
  conventions, the already-in-container check, and the available CTK × host-compiler matrix.
  Triggers: "run in devcontainer", "launch the container", "build with cuda 13.2", "open a shell with gcc 14", "start a devcontainer".
---

# cccl-devcontainer

`.devcontainer/launch.sh` boots a Docker container preconfigured with a chosen CUDA toolkit
and host compiler, mounts the repo, and either drops into a shell or runs a script.
**Linux-only** — Linux host, Linux container. Windows / MSVC builds run outside the devcontainer.

## Launch flags

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

```
.devcontainer/launch.sh -d --cuda 13.2 --host gcc14
.devcontainer/launch.sh -d --cuda 12.9 --host gcc13 -- ./ci/build_cub.sh -cxx g++ -std 17 -arch native
.devcontainer/launch.sh -d --gpus all -- ./ci/util/build_and_test_targets.sh --preset cub-cpp20 --build-targets "cub.cpp20.test.iterator"
```

For targeted builds inside a container, route to `cccl-build`; for tests, `cccl-test`.

## Already inside a container?

`CCCL_BUILD_INFIX` is set inside the container. Check before launching:

```
echo "$CCCL_BUILD_INFIX"
```

Non-empty → already inside; run the command directly. Nested launches don't work.
First launch pulls the image; subsequent launches use cache.

## Additional resources

- `references/regenerate.md` — when and how to regenerate devcontainer subdirs from `ci/matrix.yaml`.
- `references/docs.md` — index of devcontainer documentation.
- `references/tools.md` — devcontainer scripts with purpose and ownership.
- `references/launch_usage.md` — `.devcontainer/launch.sh` interface and examples.
