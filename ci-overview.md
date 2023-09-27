# Continuous Integration (CI) Overview for CCCL

The Continuous Integration (CI) process for CCCL ensures code quality and compatibility across various environments. This document provides an in-depth overview of the CI setup and workflows, enabling contributors to understand, debug, and reproduce CI checks locally.

## CI Environment and Configuration

### Development Containers

Our CI jobs use the same Development Containers as described in the [dev container setup](.devcontainer/README.md). This ensures a consistent environment for both local development and CI testing. These containers provide a standardized, reproducible environment that has all the necessary dependencies installed.

### Matrix Testing
To ensure compatibility across various setups, CI tests are performed across a broad matrix of:

- CUDA versions
- Compilers
- GPU architectures
- Operating systems

The exact combinations of these environments are defined in the [`ci/matrix.yaml`](ci/matrix.yaml) file.

### Special CI Commands

During your development, there might be scenarios where you want to control the execution of the CI pipeline based on the nature of your commits. For this, we've provided special commands that can be included in your commit messages to direct the CI:

- `[skip ci]`: By adding this command in your commit message, you signal the CI to completely skip running the pipeline for that commit. This can be handy for changes that are purely documentation-based or others that don't require CI validation.

- `[skip-tests]`: Sometimes, you might want the CI to run but skip the GPU tests. Including this command ensures that other parts of the pipeline run, but the GPU tests are bypassed. This is useful for preliminary pushes or when working on parts of the codebase that don't touch GPU functionalities.

Always ensure that you're using these commands judiciously. While they provide flexibility, they should be used appropriately to maintain the codebase's integrity and quality.

### Accelerating Build Times with `sccache`

To speed up compilation, our CI uses [`sccache`](https://github.com/mozilla/sccache), a distributed cache system that caches previously built compiler artifacts if the corresponding files haven't changed. This cache is also shared with the [Development Containers](.devcontainer/README.md). When building locally within a devcontainer, the cache is used, ensuring consistent and fast build times both locally and in CI. The shared nature of sccache provides a virtuous cycle: CI accelerates local builds by contributing to the cache, and in turn, local builds accelerate CI by populating the cache with more artifacts. This synergy ensures optimal build performance and reduces the time taken for iterative development and testing. To benefit from this shared cache, ensure you've set up [GitHub Authentication](.devcontainer/README.md#5-github-authentication) in your devcontainer.


### Build and Test Scripts

CI jobs utilize the build and test scripts located in the `ci/` directory. This ensures uniformity in how the code is built and tested, regardless of where it's being executed. When you build or test locally, you're using the same scripts as the CI, minimizing discrepancies between local and CI environments. For more detailed instructions on how to use these scripts, please refer to the [CONTRIBUTING.md guide](CONTRIBUTING.md#building-and-testing).

### Reproducing CI Failures Locally

If your pull request encounters a failure during CI testing, it's usually helpful to reproduce the issue locally to diagnose and fix it. Here's a step-by-step guide to ensure you recreate the exact environment and situation:

1. **Get the Appropriate Development Container**:

    Our CI uses the same development containers as those you'd use for local development, ensuring a consistent environment. These containers come pre-configured with all the necessary tools, libraries, and settings.

    If you aren't already doing so, make sure to use the devcontainers for local development. For details on setting up and launching the appropriate dev container, refer to the [Dev Containers guide](.devcontainer/README.md).

    The CI logs will mention the exact environment used. Ensure you launch the corresponding container locally.

2. **Run the Build/Test Script**:

    Inside the container, navigate to the root of the `cccl` project. Use the scripts from the `ci/` directory to build and test the project, just as the CI does.

    Example:
    ```bash
    ./ci/build_cub.sh <HOST_COMPILER> <CXX_STANDARD> <GPU_ARCHS>
    ./ci/test_cub.sh <HOST_COMPILER> <CXX_STANDARD> <GPU_ARCHS>
    ```

    The CI logs provide exact instructions on the scripts and parameters used, making it straightforward to reproduce the exact CI steps locally.

    Here is an example of a CI failure message that includes instructions. Note that the instructions may have changed. Refer to the latest failure log for the most up-to-date instructions.
    ![Shows an example of a CI failure log with reproducer instructions](docs/images/repro_instructions.png).


## CI Workflow Details

### Triggering Mechanism and `copy-pr-bot`

CCCL uses NVIDIA's self-hosted action runners to execute CI jobs. Due to security considerations, we use a unique approach to triggering PR workflows. Rather than triggering directly on pull request events, we utilize the [`copy-pr-bot` GitHub application](https://docs.gha-runners.nvidia.com/onboarding/). This bot streamlines the process of deeming code as trusted by copying it to a prefixed branch, ensuring that only safe and vetted code runs on our self-hosted runners.

If you're an external contributor, be aware that the CI won't start automatically. Instead, a repository member will first review your changes. After ensuring the changes meet the required criteria, they will use the `/ok to test` comment to initiate the CI process. This extra step ensures the security and integrity of our CI process.

## Troubleshooting CI Failures

1. **Check the CI logs**: Always start by examining the detailed logs provided by the CI. They will provide specific error messages that can guide your troubleshooting process.
2. **Reproduce Locally**: As previously mentioned, try to reproduce the issue in your local development environment. This allows for more rapid iteration and debugging.
3. **Matrix Configuration**: If a specific combination of CUDA version, compiler, or GPU architecture is causing the failure, consult the `ci/matrix.yaml` to understand the testing combinations.
4. **Seek Help**: If you're unable to resolve a CI failure, don't hesitate to ask. The NVIDIA team and community can provide insights or point out common pitfalls.

## Conclusion

Our CI pipeline is a living entity, continuously evolving to meet the needs of the project. While we strive to keep this documentation up-to-date, always refer to the actual code and scripts if you're in doubt.

Understanding the CI process is crucial for smooth contributions to CCCL. By ensuring your changes pass the CI checks and being able to debug any issues that arise, you streamline the contribution process, making it more efficient for both you and the maintainers. Always remember that the goal is to ensure that CCCL remains a high-quality, robust library that serves its community effectively.