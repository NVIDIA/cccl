

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


Alright, let me first examine the `.devcontainer` directory and the `ci/matrix.yaml` file to understand the role and functionality of the `make_devcontainers.sh` script. Once I have a grasp on its purpose and use, I'll suggest a section for the README that describes the script for maintainers.

After analyzing the `.devcontainer` directory and the `ci/matrix.yaml` file, I've garnered insights into how the `make_devcontainers.sh` script operates. It appears that the script generates `devcontainer` configurations for various combinations of CUDA versions and host compilers, as dictated by the matrix defined in `matrix.yaml`. The generated configurations facilitate an easy development experience across multiple environments.

Based on this understanding, here's a suggested section for the README to describe the `make_devcontainers.sh` script:


## For Maintainers: The `make_devcontainers.sh` Script

### Overview

`make_devcontainers.sh` generates devcontainer configurations for the unique combinations of CUDA Toolkit (CTK) versions and host compilers in `ci/matrix.yaml`.


### How It Works:

1. Parses the matrix from `ci/matrix.yaml`.
2. Use the top-level `.devcontainer/devcontainer.json` as a template. For each unique combination of CTK version and host compiler, generate a corresponding `devcontainer.json` configuration, adjusting only the base Docker image to match the desired environment.
3. Place the generated configurations in the `.devcontainer` directory, organizing them into subdirectories following the naming convention `cuda<CTK_VERSION>-<COMPILER_VERSION>`.

### Usage:

```bash
make_devcontainers.sh
```

**Note**: When adding or updating supported environments, modify `matrix.yaml` and then rerun this script to synchronize the `devcontainer` configurations.
