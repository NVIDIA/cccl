
# Contributing to CCCL

Thank you for your interest in contributing to the CUDA C++ Core Libraries (CCCL)!

## Getting Started

1. **Fork & Clone the Repository**:

   Fork the [CCCL GitHub Repository](https://github.com/nvidia/cccl) and clone the fork. For more information, check [GitHub's documentation on forking](https://docs.github.com/en/github/getting-started-with-github/fork-a-repo) and [cloning a repository](https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository).

2. **Set up Development Environment**:

   CCCL uses Development Containers to provide a consistent development environment for both local development and CI. Contributors are strongly encouraged to use these containers as they simplify environment setup. See the [Dev Containers guide](.devcontainer/README.md) for instructions on how to quickly get up and running using dev containers with or without VSCode.

## Making Changes

1. **Create a New Branch**:
   ```bash
   git checkout -b your-feature-branch
   ```

2. **Make Changes**.

3. **Build and Test**:

   Ensure changes don't break existing functionality by building and running tests.

   ```bash
   ./ci/build_[thrust|cub|libcudacxx].sh -cxx <HOST_COMPILER> -std <CXX_STANDARD> -arch <GPU_ARCHS>
   ./ci/test_[thrust|cub|libcudacxx].sh  -cxx <HOST_COMPILER> -std <CXX_STANDARD> -arch <GPU_ARCHS>
   ```
   For more details on building and testing, refer to the [Building and Testing](#building-and-testing) section below.

4. **Commit Changes**:
   ```bash
   git commit -m "Brief description of the change"
   ```

### Developer Guides

For more information about design and development practices for each CCCL component, refer to the following developer guides:

#### CUB
- [CUB Developer Guide](cub/docs/developer_overview.rst) - General overview of the design of CUB internals
- [CUB Test Overview](cub/docs/test_overview.rst) - Overview of how to write CUB unit tests
- [CUB Tuning Infrastructure](cub/docs/tuning.rst) - Overview of CUB's performance tuning infrastructure
- [CUB Benchmarks](cub/docs/benchmarking.rst) - Overview of CUB's performance benchmarks

#### Thrust
Coming soon!

#### libcudacxx
Coming soon!

## Building and Testing

CCCL components are header-only libraries. This means there isn't a traditional build process for the library itself. However, before submitting contributions, it's a good idea to build and run tests. Dedicated build and test scripts for each component are provided in the `ci/` directory.

### Building

Use the build scripts provided in the `ci/` directory to build tests for each component. Building tests does not require a GPU.

```bash
   ci/build_[thrust|cub|libcudacxx].sh -cxx <HOST_COMPILER> -std <CXX_STANDARD> -arch <GPU_ARCHS>

- **HOST_COMPILER**: The desired host compiler (e.g., `g++`, `clang++`).
- **CXX_STANDARD**: The C++ standard version (e.g., `11`, `14`, `17`, `20`).
- **GPU_ARCHS**: A semicolon-separated list of CUDA GPU architectures (e.g., `"70;85;90"`). This uses the same syntax as CMake's [CUDA_ARCHITECTURES](https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html#prop_tgt:CUDA_ARCHITECTURES):
   - `70` - both PTX and SASS
   - `70-real` - SASS only
   - `70-virtual` - PTX only

**Example:**
```bash
./ci/build_cub.sh -cxx g++ -std 14 -arch "70;75;80-virtual"
```

### Testing

Use the test scripts provided in the `ci/` directory to run tests for each component. These take the same arguments as the build scripts and will automatically build the tests if they haven't already been built. Running tests requires a GPU.

```bash
   ci/test_[thrust|cub|libcudacxx].sh -cxx <HOST_COMPILER> -std <CXX_STANDARD> -arch <GPU_ARCHS>
```

**Example:**
```bash
./ci/test_cub.sh -cxx g++ -std 14 -arch "70;75;80-virtual"
```

## Creating a Pull Request

1. Push changes to your fork
2. Create a pull request targeting the `main` branch of the original CCCL repository. Refer to [GitHub's documentation](https://docs.github.com/en/github/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests) for more information on creating a pull request.
3. Describe the purpose and context of the changes in the pull request description.

## Continuous Integration (CI)

CCCL's CI pipeline tests across various CUDA versions, compilers, and GPU architectures.
For external contributors, the CI pipeline will not begin until a maintainer leaves an `/ok to test` comment. For members of the NVIDIA GitHub enterprise, the CI pipeline will begin immediately.
For a detailed overview of CCCL's CI, see [ci-overview.md](ci-overview.md).

## Review Process

Once submitted, maintainers will be automatically assigned to review the pull request. They might suggest changes or improvements. Constructive feedback is a part of the collaborative process, aimed at ensuring the highest quality code.

For constructive feedback and effective communication during reviews, we recommend following [Conventional Comments](https://conventionalcomments.org/).

Further recommended reading for successful PR reviews:
- [How to Do Code Reviews Like a Human (Part One)](https://mtlynch.io/human-code-reviews-1/)
- [How to Do Code Reviews Like a Human (Part Two)](https://mtlynch.io/human-code-reviews-2/)

## Thank You!

Your contributions enhance CCCL for the entire community. We appreciate your effort and collaboration!
