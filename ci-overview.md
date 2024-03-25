# Continuous Integration (CI) Overview for CCCL

The Continuous Integration (CI) process for CCCL ensures code quality and compatibility across various environments. This document provides an in-depth overview of the CI setup and workflows, enabling contributors to understand, debug, and reproduce CI checks locally.

## CI Environment and Configuration

### Development Containers

CCCL's CI jobs use the same Development Containers as described in the [Dev Container setup](.devcontainer/README.md). Follow the instructions in that guide to set up a development container with the same environment as CI.

### Matrix Testing
To ensure compatibility across various setups, CI tests are performed across a broad matrix of:

- CUDA versions
- Compilers
- GPU architectures
- Operating systems

The exact combinations of these environments are defined in the [`ci/matrix.yaml`](ci/matrix.yaml) file.

### Viewing CI Workflow Results

The results of every job in the CI pipeline are summarized on the bottom of the PR page. Click the "Details" link next to each run to provide more detailed information.

![Summary of all CI jobs on PR page.](docs/images/pr-checks.png).

### Special CI Commands

Special commands are provided that can be included in commit messages to direct the CI pipeline execution:

- `[skip ci]`: Skips the entire CI pipeline. Useful for documentation changes or others that don't require CI validation.

   - **Example:** `git commit -m "[skip ci] Update README."`

- `[skip-tests]`: Skips CI jobs that execute tests, but runs all other jobs. Useful to avoid time-consuming tests when changes are unlikely to affect them.

Use these commands judiciously. While they offer flexibility, they should be used appropriately to maintain the codebase's integrity and quality.

### Accelerating Build Times with `sccache`

CCCL's CI uses [`sccache`](https://github.com/mozilla/sccache) to cache compiler artifacts for files that haven't changed and dramatically accelerate build times. Local builds inside [CCCL's Dev Containers](.devcontainer/README.md) can share the same cache such that local builds and CI jobs mutually benefit from accelerated build times. Follow the [GitHub Authentication](.devcontainer/README.md#optional-authenticate-with-github-for-sccache) guide to enable this feature.

### Build and Test Scripts

CI jobs employ the build and test scripts in the `ci/` directory to build and run tests. These scripts provide a consistent entry point for building and testing in both local and CI environments. For more information on using these scripts, see the [CONTRIBUTING.md guide](CONTRIBUTING.md#building-and-testing).

### Reproducing CI Failures Locally

If a pull request encounters a failure during CI testing, it is usually helpful to reproduce the issue locally to diagnose and fix it. Here is a step-by-step guide to recreate the exact environment and situation:

1. **Get the Appropriate Development Container**:

    CI jobs use the same [development containers](.devcontainer/README) as those used for local development.

    In order to simplify reproducing an issue in CI, it is recommended to use the same container locally. The CI logs will mention the exact environment used.

2. **Run the Build/Test Script**:

    CI jobs use the build and test scripts found in the `ci/` directory.

    Example:
    ```bash
    ./ci/build_cub.sh <HOST_COMPILER> <CXX_STANDARD> <GPU_ARCHS>
    ./ci/test_cub.sh <HOST_COMPILER> <CXX_STANDARD> <GPU_ARCHS>
    ```

    The CI logs provide exact instructions on the scripts and parameters used.

    Here is an example of a CI failure message that includes instructions how to clone the exact same commit and run the relevant script in the appropriate container. Note that the instructions may have changed. Refer to the latest failure log for the most up-to-date instructions.
    ![Shows an example of a CI failure log with reproducer instructions](docs/images/repro_instructions.png).

## CI Workflow Details

### Triggering Mechanism and `copy-pr-bot`

CCCL uses [NVIDIA's self-hosted action runners](https://docs.gha-runners.nvidia.com/runners/) for CI jobs. For security, PR workflows are triggered using the [`copy-pr-bot` GitHub application](https://docs.gha-runners.nvidia.com/onboarding/), which copies code to a prefixed branch to ensure only vetted code runs on the runners.

The CI pipeline will not start automatically for external contributors. A repository member will first review the changes and initiate the CI pipeline with an `/ok to test` comment.

## Troubleshooting CI Failures

1. **Review CI logs**: Examine CI logs for specific error messages (see [Viewing CI Workflow Results](#viewing-ci-workflow-results))
2. **Reproduce Locally**: Try replicating the issue locally (see [Reproducing CI Failures Locally](#reproducing-ci-failures-locally))
3. **Ask for Assistance**: If stuck, don't hesitate to reach out to the @NVIDIA/cccl team on an issue or PR, or ask a question by starting a [Discussion](https://github.com/NVIDIA/cccl/discussions).
