

# CCCL Dev Containers

## Quickstart

1. **Install Prerequisites**:
   - [Visual Studio Code](https://code.visualstudio.com/)
   - [Remote - Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

2. Click this link to automatically open VSCode and clone the CCCL repository:

   [vscode://vscode.git/clone?url=https://github.com/nvidia/cccl.git](vscode://vscode.git/clone?url=https://github.com/nvidia/cccl.git)

Alternatively:

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


Certainly. Here's the updated "Using Docker Images Directly (Manual Approach)" section:

---

## Using Docker Images Directly (Manual Approach)

While Visual Studio Code with Dev Containers provides a seamless development experience, some developers might prefer a direct Docker approach. We've made this easy with our `launch.sh` script.

Before using the script, you might want to explore the available configurations.
The `.devcontainer` directory and its subdirectories house these configurations, with each subdirectory typically named as `cuda<CTK_VERSION>-<HOST-COMPILER>`.

### Steps:

1. **Use the `launch.sh` Script**:
   To initiate your desired development environment, run the `launch.sh` script with flags specifying your desired CUDA and compiler versions:
   ```bash
   launch.sh --cuda 12.2 --host gcc12 --docker
   ```

   This script will pull the appropriate Docker image and run it, providing you with an interactive shell inside the container.

2. **Workspace Setup**:
   The `launch.sh` script mounts the CCCL repository directory into the container by default, ensuring that your workspace inside the container mirrors your local setup.

3. **Working Inside the Container**:
   Inside the container, you have all the tools and configurations tailored for CCCL development.
   You can build and run test, or perform any other development tasks as if you were in a local environment.

4. **Exiting the Container**:
   Once you're done, you can exit the container by simply typing `exit` in the interactive shell. The container will be removed upon exit, ensuring a clean state for your next session.

