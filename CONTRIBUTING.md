# Contributing to CCCL

First and foremost, thank you for your interest in contributing to the CUDA C++ Libraries (CCCL)! This guide aims to provide a clear overview to help you get started with your contributions.

## Getting Started

### 1. **Fork the Repository & Add Original Repository as a Remote**

#### Using GitHub CLI:

If you haven't cloned the `cccl` repository yet, you can do so and set up the fork simultaneously:

```bash
gh repo clone nvidia/cccl
cd cccl
gh repo fork --remote
```

#### Manual Way:

- Navigate to the [CCCL GitHub Repository](https://github.com/nvidia/cccl).
- In the top-right corner of the page, click "Fork".

This will create a copy of the repository in your own GitHub account. This is your personal space to work on the code without affecting the main project.

After forking, on your machine, clone your forked repository. Replace `YOUR_USERNAME` with your GitHub username:

```bash
git clone https://github.com/YOUR_USERNAME/cccl.git
cd cccl
```

To keep track of changes in the original repository, add it as a remote named `upstream`:

```bash
git remote add upstream https://github.com/nvidia/cccl.git
```


This sequence of commands will clone the `cccl` repository, navigate into the directory, and then create a fork in your GitHub account while also setting up the `upstream` remote to point to the original `cccl` repository.

#### Verify:

To verify that your remotes have been set up correctly, use:
```bash
git remote -v
```

You should see something like:
```bash
origin	git@github.com:YOUR_USERNAME/cccl.git (fetch)
origin	git@github.com:YOUR_USERNAME/cccl.git (push)
upstream	git@github.com:NVIDIA/cccl.git (fetch)
upstream	git@github.com:NVIDIA/cccl.git (push)
``````

### 2. **Set up Development Environment**:

CCCL uses Development Containers to provide a consistent development environment for both local development and CI.
Contributors are strongly encouraged to use these containers as they dramatically simplify setting up your environment.
See the [Dev Containers guide](.devcontainer/README.md) for instructions on how to quickly get up and running using dev containers with or without VSCode.

## Making Changes

1. **Create a New Branch**:

    Always create a new branch for your changes:

    ```bash
    git checkout -b your-feature-branch
    ```

2. **Write Code**:

    Make your changes in the code.

3. **Build and Test**:

    Before committing, ensure your changes don't introduce regressions by building and running tests:

    ```bash
    ./ci/build_COMPONENT.sh <HOST_COMPILER> <CXX_STANDARD> <GPU_ARCHS>
    ./ci/test_COMPONENT.sh  <HOST_COMPILER> <CXX_STANDARD> <GPU_ARCHS>
    ```

    For more details on building and testing, refer to the [Building and Testing](#building-and-testing) section below.

4. **Commit Your Changes**:

    Group your changes and commit them with a meaningful commit message:

    ```bash
    git commit -m "Add a brief description of the change"
    ```
### Developer Guides

As CCCL evolves, we are in the process of unifying and consolidating our practices and guidelines across all components. However, due to the unique history and characteristics of each component, there are still aspects specific to each one. To aid your contributions and ensure you're aligned with each component's best practices, we offer detailed developer guides:

#### CUB
- [CUB Developer Guide](cub/docs/developer_overview.rst) - General overview of the design of CUB internals
- [CUB Test Overview](cub/docs/test_overview.rst) - Overview of how to write CUB unit tets
- [CUB Tuning Infrastructure](cub/docs/tuning.rst) - Overview of CUB's performance tuning infrastructure
- [CUB Benchmarks](cub/docs/benchmarking.rst) - Overview of CUB's performance benchmarks

#### Thrust

Coming soon!

#### libcudacxx

Coming soon!



These guides delve deeper into the specifics of each library and will be invaluable if you're looking to make substantial changes or additions.

## Building and Testing

CCCL components are header-only libraries. This means there isn't a traditional build process for the library itself. However, before submitting your contributions, it's crucial to build and test your changes against the suite of tests. Dedicated build and test scripts for each component are provided in the `ci/` directory to facilitate this.

### Building

To build the tests, you can use the respective build scripts. Building tests does not require a GPU.

Each script expects the following arguments:

- **HOST_COMPILER**: The host compiler you wish to use (e.g., `g++`, `clang++`).
- **CXX_STANDARD**: The C++ standard version (e.g., `11`, `14`, `17`, `20`).
- **GPU_ARCHS**: A semicolon-separated list of CUDA GPU architectures (e.g., `70;85;90`). This uses the same syntax as CMake's [CUDA_ARCHITECTURES](https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html#prop_tgt:CUDA_ARCHITECTURES):
   - `70` - both PTX and SASS
   - `70-real` - SASS only
   - `70-virtual` - PTX only

- **CUB**:
    ```bash
    ./ci/build_cub.sh <HOST_COMPILER> <CXX_STANDARD> <GPU_ARCHS>
    ```

- **libcudacxx**:
    ```bash
    ./ci/build_libcudacxx.sh <HOST_COMPILER> <CXX_STANDARD> <GPU_ARCHS>
    ```

- **Thrust**:
    ```bash
    ./ci/build_thrust.sh <HOST_COMPILER> <CXX_STANDARD> <GPU_ARCHS>
    ```

    Example:
    ```bash
    ./ci/build_cub.sh g++ 14 "70;75;80-virtual"
    ```

### Testing the Components

To execute the tests, you can use similar test scripts. They take the same arguments as the build scripts. The test scripts will automatically build the tests if they haven't already been built.
Running tests requires a GPU:

- **CUB**:
    ```bash
    ./ci/test_cub.sh <HOST_COMPILER> <CXX_STANDARD> <GPU_ARCHS>
    ```

- **libcudacxx**:
    ```bash
    ./ci/test_libcudacxx.sh <HOST_COMPILER> <CXX_STANDARD> <GPU_ARCHS>
    ```

- **Thrust**:
    ```bash
    ./ci/test_thrust.sh <HOST_COMPILER> <CXX_STANDARD> <GPU_ARCHS>
    ```

Example:
```bash
./ci/test_cub.sh g++ 14 "70;75;80-virtual"
```

## Submitting Your Changes

1. **Push to Your Fork**:

   Push your branch to your fork:

   ```bash
   git push origin your-feature-branch
   ```

2. **Open a Pull Request**:

   You can create a pull request using the GitHub website or directly using the `gh` CLI:

   - **Using GitHub**:

     Go to the main page of the `cccl` GitHub repo. Click the "New pull request" button. Ensure the "base" branch is `main` and the "compare" branch is `your-feature-branch`.

   - **Using `gh` CLI**:

     ```bash
     gh pr create --base main --head your-feature-branch --repo nvidia/cccl
     ```
     Follow the interactive prompts to provide a title and a body for your pull request.

3. **Describe Your Changes**:

   Whether you're using the GitHub website or `gh` CLI, provide a concise description of what the PR does, why you're proposing these changes, and any other relevant details.

## Continuous Integration (CI)

CCCL has a comprehensive CI pipeline that tests across various CUDA versions, compilers, and GPU architectures. All PRs must pass our CI before they can be merged.

For external contributors, the CI pipeline will not begin until a maintainer leaves a `/ok to test` comment. For members of the NVIDIA GitHub enterprise, the CI pipeline will begin immediately.

For a detailed understanding of our CI process, refer to the [ci-overview.md](ci-overview.md) document.

## Review Process

Once you've submitted a pull request, it will be reviewed by the maintainers. They might suggest some changes, improvements, or alternatives.

Remember, code review is a part of the collaborative process, and making required changes will ensure your contributions get merged smoothly.

To make the review process as easy as possible for everyone, we encourage review comments to follow [Conventional Comments](https://conventionalcomments.org/).

Further recommended reading for sucessful PR reviews:
- [How to Do Code Reviews Like a Human (Part One)](https://mtlynch.io/human-code-reviews-1/)
- [How to Do Code Reviews Like a Human (Part Two)](https://mtlynch.io/human-code-reviews-2/)

## Thank You!

Your contributions help improve CCCL for everyone. We appreciate your effort and look forward to collaborating with you!

