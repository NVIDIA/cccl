> **Note**
> The instructions in this README are specific to Linux development environments. Instructions for Windows are coming soon!

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/NVIDIA/cccl?quickstart=1&devcontainer_path=.devcontainer%2Fdevcontainer.json)

# CCCL Dev Containers

CCCL uses [Development Containers](https://containers.dev/) to provide consistent and convenient development environments for both local development and for CI. This guide covers setup in [Visual Studio Code](#quickstart-vscode-recommended) and [Docker](#quickstart-docker-manual-approach).

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

Without authentication to the remote server, `sccache` will still accelerate local builds by using a filesystem cache.

Follow the instructions in the prompt as below and enter the one-time code at https://github.com/login/device

  ![Shows authentication with GitHub to access sccache bucket.](./img/github_auth.png)

To manually trigger this authentication, execute the `devcontainer-utils-vault-s3-init` script within the container.

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
    This script starts an interactive shell as the `coder` user inside the container with the local `cccl/` directory mirrored into `/home/coder/cccl`.

    For specific environments, use the `--cuda` and `--host` options:
    ```bassh
    ./.devcontainer/launch.sh --docker --cuda 12.2 --host gcc10
    ```
    See `./.devcontainer/launch.sh --help` for more information.

2. Done. See the [contributing guide](../CONTRIBUTING.md#building-and-testing) for instructions on how to build and run tests.

## Available Environments

CCCL provides environments for both the oldest and newest supported CUDA versions with all compatible host compilers.

Look in the [`.devcontainer/`](.) directory to see the available configurations. The top-level [`devcontainer.json`](./devcontainer.json) serves as the default environment. All `devcontainer.json` files in the `cuda<CTK_VERSION>-<HOST-COMPILER>` sub-directories are variations on this top-level file, with different base images for the different CUDA and host compiler versions.

## VSCode Customization

By default, CCCL's Dev Containers come with certain VSCode settings and extensions configured by default, as can be seen in the [`devcontainer.json`](./devcontainer.json) file. This can be further customized by users without needing to modify the `devcontainer.json` file directly.

For extensions, the [`dev.containers.defaultExtensions` setting](https://code.visualstudio.com/docs/devcontainers/containers#_always-installed-extensions) allows listing extensions that will always be installed.

For more general customizations, VSCode allows using a dotfile repository. See the [VSCode documentation](https://code.visualstudio.com/docs/devcontainers/containers#_personalizing-with-dotfile-repositories) for more information.

## GitHub Codespaces

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/NVIDIA/cccl?quickstart=1&devcontainer_path=.devcontainer%2Fdevcontainer.json)

One of the benefits of Dev Containers is that they integrate natively with [GitHub Codespaces](https://github.com/features/codespaces). Codespaces provide a VSCode development environment right in your browser running on a machine in the cloud. This provides a truly one-click, turnkey development environment where you can develop, build, and test with no other setup required.

Click the badge above or [click here](https://codespaces.new/NVIDIA/cccl?quickstart=1&devcontainer_path=.devcontainer%2Fdevcontainer.json) to get started with CCCL's Dev Containers on Codespaces. This will start the default Dev Container environment. [Click here](https://github.com/codespaces/new?hide_repo_select=true&ref=main&repo=296416761&skip_quickstart=true) to start a Codespace with a particular environment and hardware configuration as shown:

   ![Shows configuring a Codespace with a custom environment](../docs/images/codespaces.png)

## For Maintainers: The `make_devcontainers.sh` Script

### Overview

[`make_devcontainers.sh`](./make_devcontainers.sh) generates devcontainer configurations for the unique combinations of CUDA Toolkit (CTK) versions and host compilers in [`ci/matrix.yaml`](../ci/matrix.yaml).

### How It Works:

1. Parses the matrix from `ci/matrix.yaml`.
2. Use the top-level [`.devcontainer/devcontainer.json`](./devcontainer.json) as a template. For each unique combination of CTK version and host compiler, generate a corresponding `devcontainer.json` configuration, adjusting only the base Docker image to match the desired environment.
3. Place the generated configurations in the `.devcontainer` directory, organizing them into subdirectories following the naming convention `cuda<CTK_VERSION>-<COMPILER_VERSION>`.

For more information, see the `.devcontainer/make_devcontainers.sh --help` message.

**Note**: When adding or updating supported environments, modify `matrix.yaml` and then rerun this script to synchronize the `devcontainer` configurations.
