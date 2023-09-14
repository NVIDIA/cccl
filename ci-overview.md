# CCCL - Continuous Integration (CI) Workflow 
As a header-only C++ library, the code we write isn't compiled until a developer includes it in their project. 

This means our code needs to be robust enough to compile and run across a variety of platforms, compilers, and configurations. 
As such, maintaining extensive and thorough Continuous Integration (CI) coverage is crucial. 

Our CI system needs to test various combinations of operating systems, CPU architectures, compilers, and C++ standards. 
The number of configurations can be quite large and hence our CI system must be designed to handle this complexity. 
We use GitHub Actions for this purpose. It is flexible, powerful, and well-integrated with GitHub.

This document will walk you through our CI workflow and guide you on how to interact with and troubleshoot it when needed.

## Workflow Overview

### TL;DR

```mermaid
graph LR
    A[Workflow Starts on Push to PR]
    A --> B{Computes Matrix}
    B --> C[For Each Matrix Configuration]
    C --> D{Runs a Build Job}
    D --> E[Build Job Completed]
    E --> F{Runs a Test Job}
    F --> G[Test Job Completed]
    G --> H[CI Successful]
```

This repository relies on a GitHub Actions-based Continuous Integration (CI) workflow. Here's what you need to know:

- **Trigger:** The main workflow triggers on every push to the main branch or pull request (PR).
- **Execution:** The workflow generates a matrix of build configurations, based on the settings in matrix.yml, and then dispatches separate build and test jobs for each configuration. 
- **Failures:** If a job fails, you'll be notified through GitHub's interface. You can then check the logs for details.
- **Recovery:** To handle job failures, pull the relevant container image and rerun the script locally to reproduce the issue.

### The Matrix

The matrix defined in the [`matrix.yml`](ci/matrix.yaml) is the single source of truth for the environments we test our code against.
It dictates the build configurations, such as CUDA version, operating system, CPU architecture, compiler, GPU architectures, and C++ standards. 
It allows us to test our code against different combinations of these variables to ensure our code's compatibility and stability. 

### Build and Test Jobs
Our CI workflow primarily revolves around two major types of jobs: build jobs and test jobs. 

#### Build
Build jobs compile our unit tests and examples in various environments, mimicking the conditions in which our users might compile the code. 
These jobs simply invoke a build script (e.g., `build_thrust.sh`) which contains all the necessary steps to compile the code.

The advantage of this approach is two-fold. 
First, it allows us to keep our CI configuration files clean and focused on the orchestration of jobs rather than the specifics of building the code. 
Second, it greatly simplifies the process of reproducing build issues outside of CI. 
Developers can run the build script locally, in their environment, and the build will behave in the same way it does in CI.

#### Test
After the build jobs have successfully compiled the test binaries, the test jobs run these binaries to execute our tests. 
On first glance, you may notice that the test jobs are rebuilding all of the test binaries. 
However, we are relying on sccache to cache the build artifacts from the build jobs and reuse them in the test jobs.

Similar to the build jobs, test jobs use a script (e.g., `test_thrust.sh`) to define the steps required to execute the tests.
If a test fails in CI, developers can simply run the script in their local environment to reproduce the issue and debug it.

The syntax of the build and test scripts is the same:
```bash
./ci/build_thrust.sh <host compiler> <c++ standard> <gpu architectures>
./ci/test_thrust.sh <host compiler> <c++ standard> <gpu architectures>

#examples
./ci/build_thrust.sh g++ 17 70;80;86
```

In summary, the heart of our build and test jobs is the corresponding build or test script. 
This design philosophy helps maintain a clear separation between CI orchestration and the specifics of building and testing. 
Moreover, it paves the way for straightforward issue reproduction, thereby aiding developers in their debugging process.

## Lifecycle of a Pull Request

From creation to merging, a pull request in this project follows these steps:

1. Create a PR: Once you make a change, open a PR. 
    - If you have write permission to the repository, CI workflow will automatically start.
    - If you don't have write permission, the workflow will start once a maintainer comments on the PR with `/ok to test`. This comment is required for all subsequent workflow runs.
2. Wait for results: GitHub Actions executes the defined CI workflow, running jobs based on the matrix configuration.
3. Interpret results: Check the status of the workflow. If it passes, all tests have passed on all defined configurations, and your changes likely didn't break anything. 
4. Handle failures: If any job fails, the logs will provide information on what went wrong. 
5. Rerun jobs: If the failure seems unrelated to your changes (e.g., due to a temporary external issue), you can rerun the jobs.

## Troubleshooting Guide

If a CI job fails, here's what you can do to troubleshoot:

1. Check the logs: The logs provide detailed information on what went wrong during the execution. This is your starting point.
2. Reproduce the issue locally: Pull the relevant container image and rerun the script that failed. This will allow you to dig into the issue in depth.
3. Fix the issue: Once you've identified the problem, you can make appropriate changes to your code and rerun the CI jobs.

### How to Reproduce a CI Failure Locally

When a build or test job fails, it will provide instructions on how to reproduce the failure locally using the exact same code and environment used in CI.

For example, here is a screenshot of the log of a failed build job:

![Build Job Failure](docs/images/repro_instructions.png)

This provides instructions for both a command-line and a VSCode-based approach to reproduce the failure locally.

When interating on a fix, the vscode devcontainer approach is recommended as it provides a convenient, interactive environment to debug the issue.

## More Information

You can refer to [GitHub Actions documentation](https://docs.github.com/en/actions) for a deeper understanding of the process and the [GitHub Actions workflows syntax](https://docs.github.com/en/actions/reference/workflow-syntax-for-github-actions) to comprehend the workflow files' structure and syntax.

You're not in this alone - the community is here to help. If you're stuck, don't hesitate to raise an issue or ask for assistance. Open source thrives on collaboration and learning. Happy coding!

