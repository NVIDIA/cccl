> **Note**
> Platform Note: The instructions in this README are tailored for Linux environments. If you're using Windows, please be patient as we're actively working on adding support and instructions for Windows development environments. Stay tuned!

# CCCL Dev Containers

To ensure consistency and ease of setup, the CUDA C++ Libraries (CCCL) team uses [Development Containers](https://code.visualstudio.com/docs/devcontainers/containers).
This guide offers information on setting up container in Visual Studio Code and manually with Docker.

## Quickstart: VSCode (Recommended)

### **Prerequisites**:
- [Visual Studio Code](https://code.visualstudio.com/)
- [Remote - Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

### **Step 1: Clone the Repository**

- **Easy Way**:
Copy and paste the following URI into your browser to automatically clone and open in VSCode:

    ```
    vscode://vscode.git/clone?url=https://github.com/nvidia/cccl.git
    ```

- **Manual Way**:
  ```bash
  git clone https://github.com/nvidia/cccl.git
  ```

### **Step 2: Launch Development Environment in VSCode**

- Open the cloned directory in VSCode.
- A prompt will appear in the lower right corner suggesting to "Reopen in Container". Click it.
- A list of devcontainers will be shown. These are named with a pattern `cuda<CTK_VERSION>-<HOST-COMPILER>`. Select the desired environment to start coding!

## Quickstart: Docker (Manual Approach)

### **Prerequisites**:
- [Docker](https://docs.docker.com/desktop/install/linux-install/)


### **Step 1: Clone the Repository**
```bash
git clone https://github.com/nvidia/cccl.git
```

### **Step 2: Launch Docker Container**
Navigate to the cloned directory and use the `launch.sh` script:

```bash
cd cccl
./launch.sh --docker
```

This script initiates a Docker container tailored for the specified CUDA and compiler versions. Inside, you'll get an interactive shell session with the `cccl` repository already mounted, allowing direct editing and testing.

For a deeper dive into dev containers, configurations, and benefits, continue reading below.

## Overview

The CUDA C++ Libraries (CCCL) team uses [Development Containers, or "Dev Containers"](https://code.visualstudio.com/docs/devcontainers/containers), to provide a consistent, reproducible, and isolated development environment shared for both local development and CI.
We encourage external contributors to use them as they provide an effortless onramp to setting up your environment for contributions. With just a few actions in Visual Studio Code or a basic Docker command, you're ready in the same environment the CCCL team works in.

If Docker feels daunting, Dev Containers make it more intuitive. They manage Docker's intricacies, freeing you to code. No need for complex Docker commands or profound containerization knowledge. The `devcontainer.json` file dictates each environment, designating a base Docker image with the desired CUDA Toolkit (CTK) and host compiler version. This file might also detail extra configurations, tools, or settings for further refinement.

### Available Dev Container Configurations:

A frequent challenge in CUDA development is managing different CUDA toolkit and compiler versions. Dev Containers address this by presenting environments tailored to numerous CTK and compiler combinations. By default, the CCCL team provides configurations for both the oldest and newest supported CUDA versions, with every supported host compiler for those CUDA versions.

To see the configurations available, look through the [`.devcontainer` directory](.). Each environment resides in its folder, named in the format `cuda<CTK_VERSION>-<HOST-COMPILER>`, with the related `devcontainer.json` explaining its specs. All Dev Container environments are fundamentally identical, only differing in the CUDA Toolkit (CTK) and host compiler versions. This uniformity guarantees that all developers, either from the CCCL team or external contributors, work within a standardized environment, reducing discrepancies from diverse local setups.


## Using Dev Containers with VSCode

Using Development Containers in Visual Studio Code provides a seamless integration, allowing developers to work within a containerized environment directly from the editor.

### **1. Open the Cloned Directory in VSCode**

After you've cloned the repository, navigate to the directory in your file system and open it using Visual Studio Code.

### **2. Launching a Dev Container**

There are multiple ways to launch a Dev Container in VSCode:

**a. "Reopen in Container" Prompt**:
If you have the "Remote - Containers" extension installed, a prompt will appear at the lower right corner suggesting to "Reopen in Container" once the directory is opened in VSCode.

![Shows "Reopen in Container" prompt when opening the cccl directory in VScode.](./img/reopen_in_container.png)

**b. Command Palette**:
You can also use the Command Palette to start a Dev Container. Press `Ctrl+Shift+P` to open the Command Palette. Type "Remote-Containers: Reopen in Container" and select it.

![Shows "Reopen in Container" in command pallete.](./img/open_in_container_manual.png)

**d. Selecting a Devcontainer**:
Upon choosing any of the above methods, you'll be presented with a list of available devcontainers. They're named following a pattern indicating their configurations, such as `cuda<CTK_VERSION>-<HOST-COMPILER>`. Select the desired environment from this list.

![Shows list of available container environments.](./img/container_list.png)

### **3. Initialization and Workspace Integration**

When you select a Dev Container configuration, VSCode starts the Docker image, automatically pulling it if not already present. This may take a few minutes if it needs to download the container image.

Post initialization, your local `cccl` repository is mounted into the container. This setup offers an integrated workspace where changes in VSCode directly reflect inside the container and vice-versa.

### **4. GitHub Authentication**

Shortly after starting, the Dev Container will prompt you to authenticate with GitHub. While optional, this authentication grants access to NVIDIA's distributed `sccache`, significantly accelerating builds. However, this feature is currently exclusive to NVIDIA employees who are members of the `NVIDIA` or `rapidsai` GitHub organizations.

Follow the prompt, which will have you enter a one-time code at https://github.com/login/device. Follow the instructions on the GitHub web page and after succesfully authenticating, your GitHub account is used to authenticate access to the sccache distributed storage. Succesfully completing this process looks like the following:

![Shows authentication with GitHub to access sccache bucket.](./img/github_auth.png)

This step is entirely optional and will not impact any functionality other than not being able to read or write to the distributed sccache storage. `sccache` will still write a local cache to your filesystem that will still accelerate local rebuilds.

If at any time you want to run this authentication process manually, you can do so by running the `devcontainer-utils-vault-s3-init` script inside the container.

For more information about the sccache configuration and authentication, see the documentation at [`rapidsai/devcontainers`](https://github.com/rapidsai/devcontainers/blob/branch-23.10/USAGE.md#build-caching-with-sccache).

### **5. Working Inside the Devcontainer**

With the container launched, you're now in an environment tailored for CCCL development. Whether coding, running tests, or conducting other tasks, you'll benefit from the tailored tools and settings of this environment.

For more information on building and running tests, see the [contributer guide](../CONTRIBUTING.md).

## Using Docker Images Directly (Manual Approach)

While Visual Studio Code with Dev Containers offers a smooth development experience, there might be developers who lean towards a direct Docker approach. For those, we've simplified the process with our `launch.sh` script.

Before using the script, you might want to explore the available configurations.
The `.devcontainer` directory and its subdirectories house these configurations, with each subdirectory typically named as `cuda<CTK_VERSION>-<HOST-COMPILER>`.

### **Steps**:

1. **Use the `launch.sh` Script**:
   To initiate your desired development environment, run the `launch.sh` script with flags specifying your desired CUDA and compiler versions:
   ```bash
   ./launch.sh --cuda 12.2 --host gcc12 --docker
   ```
   This command pulls the appropriate Docker image, runs it, and provides an interactive shell within the container.

2. **Workspace Setup**:
   The `launch.sh` script automatically mounts the CCCL repository directory into the container. This ensures you have a mirrored workspace inside the container, resembling your local setup.

3. **Working Inside the Container**:
   Inside the container, you are equipped with all tools and configurations designed for CCCL development. You can initiate builds, run tests, or perform any other development tasks as you would in a local setup.

4. **Exiting the Container**:
   After you're finished with your tasks, simply type `exit` in the interactive shell to leave. Upon exit, the container gets removed, assuring a pristine state for your next session.

   Certainly! Here's a minimal documentation for the script:

## `launch.sh`: Development Container Launcher Script

### Overview

This script provides a streamlined process to launch a development container environment tailored for CCCL development.
The environment can be started either in Docker directly or within Visual Studio Code (VSCode) using its "Remote - Containers" feature.

### Usage

```bash
./launch.sh [-c|--cuda <CUDA version>] [-H|--host <Host compiler>] [-d|--docker]
```

### Options

- `-c, --cuda`: Specify the CUDA version for the environment. For example, `12.2`.
- `-H, --host`: Specify the host compiler version for the environment. For example, `gcc12`.
- `-d, --docker`: By default, the script assumes you want to launch the environment in VSCode. Use this option if you want to run the development environment in Docker directly.
- `-h, --help`: Display the help message.

### Examples

1. Launch the default devcontainer in VSCode:
   ```bash
   ./launch.sh
   ```

2. Launch a specific CUDA version (`11.2`) with a specific host compiler (`gcc10`) in VSCode:
   ```bash
   ./launch.sh --cuda 11.1 --host gcc10
   ```

3. Launch the default devcontainer directly in Docker:
   ```bash
   ./launch.sh --docker
   ```

4. Launch a specific CUDA version (`11.2`) with a specific host compiler (`gcc10`) directly in Docker:
   ```bash
   ./launch.sh --cuda 11.2 --host gcc10 --docker
   ```

### Details

The script primarily works by identifying the correct `devcontainer.json` configuration based on provided arguments, then either launches a Docker container or starts a VSCode environment using that configuration. The configurations are located in `.devcontainer` directory and subdirectories.

The default behavior, when no options are provided, is to use the top-level `devcontainer.json` file in `.devcontainer` directory and launch the environment in VSCode.

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
