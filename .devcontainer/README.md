> **Note**
> The instructions in this README are specific to Linux development environments. Instructions for Windows is coming soon!

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/NVIDIA/cccl?quickstart=1&devcontainer_path=.devcontainer%2Fdevcontainer.json)

# CCCL Dev Containers

To ensure consistency and ease of setup, CCCL uses [Development Containers](https://code.visualstudio.com/docs/devcontainers/containers) for local development and for CI. This guide covers setup in Visual Studio Code and Docker.

## Quickstart: VSCode (Recommended)

### Prerequisites
- [Visual Studio Code](https://code.visualstudio.com/)
- [Remote - Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

### Steps

1. Clone the Repository
    ```bash
    git clone https://github.com/nvidia/cccl.git
    ```
2. Open the cloned directory in VSCode

3. Launch a Dev Container by clicking the prompt suggesting to "Reopen in Container"

   ![Shows "Reopen in Container" prompt when opening the cccl directory in VScode.](./img/reopen_in_container.png)

   - Alternatively, use the Command Palette to start a Dev Container. Press `Ctrl+Shift+P` to open the Command Palette. Type "Remote-Containers: Reopen in Container" and select it.

      ![Shows "Reopen in Container" in command pallete.](./img/open_in_container_manual.png)

4. Select an environment with the desired CTK and host compiler from the list:

   ![Shows list of available container environments.](./img/container_list.png)

5. VSCode will initialize the selected Dev Container. This can take a few minutes the first time.

6. Once initialized, the local `cccl/` directory is mirrored into the container to ensure any changes are persistent.

7. Done! See the [contributing guide](../CONTRIBUTING.md#building-and-testing) for instructions on how to build and run tests.

### (Optional) Authenticate with GitHub for `sccache`

After starting the container, there will be a prompt to authenticate with GitHub. This grants access to a [`sccache`](https://github.com/mozilla/sccache) server shared with CI and greatly accelerates local build times. This is currently limited to NVIDIA employees belonging to the `NVIDIA` or `rapidsai` GitHub organizations.

Follow the instructions in the prompt as below and enter the one-time code at https://github.com/login/device

  ![Shows authentication with GitHub to access sccache bucket.](./img/github_auth.png)

Even without this step, `sccache` will utilize a local cache on your filesystem, benefiting local rebuilds. To manually trigger this authentication, execute the `devcontainer-utils-vault-s3-init` script within the container.

For more information about the sccache configuration and authentication, see the documentation at [`rapidsai/devcontainers`](https://github.com/rapidsai/devcontainers/blob/branch-23.10/USAGE.md#build-caching-with-sccache).

## Quickstart: Docker (Manual Approach)

### Prerequisites
- [Docker](https://docs.docker.com/desktop/install/linux-install/)

### Steps
1. Clone the repository and use the [`launch.sh`](./launch.sh) script to launch the default container environment
    ```bash
    git clone https://github.com/nvidia/cccl.git
    cd cccl
    ./.devcontainer/launch.sh --docker
    ```
    This script starts an interactive shell inside the container with the local `cccl/` directory mirrored.

    For specific environments, use the `--cuda` and `--host` options:
    ```bassh
    ./.devcontainer/launch.sh --docker --cuda 12.2 --host gcc10
    ```
    See `./.devcontainer/launch.sh --help` for more information.

2. Done.

## Available Environments

CCCL provides environments for both the oldest and newest supported CUDA versions with all compatible host compilers.

Look in the [`.devcontainer/`](.) directory to see the available configurations.The top-level [`devcontainer.json`](./devcontainer.json) serves as the default environment. All `devcontainer.json` files in the `cuda<CTK_VERSION>-<HOST-COMPILER>` sub-directories are derived from this top-level file, just with different base images to for the different CUDA and host compiler versions.

## GitHub Codespaces

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/NVIDIA/cccl?quickstart=1&devcontainer_path=.devcontainer%2Fdevcontainer.json)

Dev Containers integrate natively with [GitHub Codespaces](https://github.com/features/codespaces) that provide a VSCode development environment right in your browser running on a machine in the cloud. This provides a truly one-click, turnkey development environment with no other setup required.

Click the badge above or [click here](https://codespaces.new/NVIDIA/cccl?quickstart=1&devcontainer_path=.devcontainer%2Fdevcontainer.json) to get started with CCCL's Dev Containers on Codespaces.

## For Maintainers: The `make_devcontainers.sh` Script

### Overview

[`make_devcontainers.sh`](./make_devcontainers.sh) generates devcontainer configurations for the unique combinations of CUDA Toolkit (CTK) versions and host compilers in [`ci/matrix.yaml`](../ci/matrix.yaml).

### How It Works:

1. Parses the matrix from `ci/matrix.yaml`.
2. Use the top-level [`.devcontainer/devcontainer.json`](./devcontainer.json) as a template. For each unique combination of CTK version and host compiler, generate a corresponding `devcontainer.json` configuration, adjusting only the base Docker image to match the desired environment.
3. Place the generated configurations in the `.devcontainer` directory, organizing them into subdirectories following the naming convention `cuda<CTK_VERSION>-<COMPILER_VERSION>`.

### Usage:
```bash
make_devcontainers.sh
```

**Note**: When adding or updating supported environments, modify `matrix.yaml` and then rerun this script to synchronize the `devcontainer` configurations.
