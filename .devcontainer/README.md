

# CCCL Dev Containers

## Quickstart

1. **Install Prerequisites**:
   - [Visual Studio Code](https://code.visualstudio.com/)
   - [Remote - Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

2. **Clone the Repository**:
   ```
   git clone https://github.com/nvidia/cccl.git
   ```
3. **Open in VSCode**: Navigate to the cloned directory and open with VSCode.

4. **Select and Launch Devcontainer**: Upon detecting a `.devcontainer` configuration, VSCode will prompt you to reopen the project inside the container. Choose your desired environment configuration and get coding!

For a deeper dive into dev containers, available configurations, and further benefits, continue reading below.

## Overview

The CUDA C++ Libraries (CCCL) team uses [Development Containers, or "Dev Containers"](https://code.visualstudio.com/docs/devcontainers/containers), to provide a consistent, reproducible, and isolated development environment shared for both local development and CI.
We encourage external contributors to use them as well as they provide an effortless onramp to setting up your environment to make contributions.
With just a few clicks in Visual Studio Code or a basic Docker command, you can immerse yourself in the same environment that the CCCL team relies on.

If Docker seems unfamiliar or intimidating, Dev Containers make it approachable.
They encapsulate Docker's complexities, letting you focus on coding. There's no need for intricate Docker commands or deep knowledge of containerization.
Instead, the devcontainer.json file defines each environment, pointing to a base Docker image equipped with the desired CUDA Toolkit (CTK) and host compiler version.
This file may also specify additional configurations, tools, or settings for further refinement.

### Available Dev Container Configurations:

Managing different versions of the CUDA toolkit and compilers is a common challenge in CUDA development.
Dev Containers simplify this aspect by offering environments tailored to various CTK and compiler combinations.
By default, the CCCL team provides configurations for both the oldest and newest supported CUDA versions, paired with all respective supported host compilers.
This approach ensures a uniform development experience.

To explore the configurations we offer, browse the [`.devcontainer` directory](.).
Each environment is encapsulated in a folder, with the associated `devcontainer.json` detailing its configuration.
All our Dev Container environments are fundamentally the same, with variations only in the CUDA Toolkit (CTK) and host compiler versions.
This consistency ensures that all developers, whether part of the CCCL team or external contributors, operate within a standardized setting, minimizing potential discrepancies stemming from varied local setups.


## What are Dev Containers?

[Dev Containers](https://containers.dev/), short for "Development Containers", make it easy to use a Docker container as a turn-key development environment for maintainers and new contributors alike.
Dev Containers ensure a consistent environment for CI and developers eliminating the "it works on my machine" problem.

## Why are Dev Containers Useful?

- **Convenience**: Provides a complete development environment at the click of a button and eliminates the arduous task of setting up an environment with all the necesserary dependencies
- **Consistency**: Every developer works in the same environment, eliminating configuration issues
- **Isolation**: Dependencies and configurations are isolated from the host machine, keeping your machine clean.
- **Portability**: Easily share and replicate development environments across teams and CI/CD processes.
- **Flexibility**: Easily switch between different environments (like different host compilers and CUDA versions).

## Structure of `.devcontainer` in this Project

Our `.devcontainer` directory contains configurations for various development environments, primarily focused on different versions of CUDA and compilers:

- Subdirectories like `cuda11.1-gcc6`, `cuda11.1-gcc7`, etc., represent different environment setups.
- `devcontainer.json` is the main configuration file that dictates how the development environment should be set up.
- `launch.sh` and `make_devcontainers.sh` are utility scripts that assist in setting up and managing these environments.

### How to Use Devcontainers in this Project

1. Install [Visual Studio Code](https://code.visualstudio.com/).
2. Install the [Remote - Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension.
3. Open this project in VS Code.
4. Press `F1` and select the "Remote-Containers: Open Folder in Container..." command.
5. Choose the appropriate environment from the `.devcontainer` directory.
6. Wait for the environment to build and start. Once done, you'll be inside the container with all dependencies installed.

**Note**: Before switching between environments, it's recommended to close the current container to avoid conflicts.

