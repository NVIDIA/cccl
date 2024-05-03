
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

CCCL components are header-only libraries. This means there isn't a traditional build process for the library itself. However, before submitting contributions, it's a good idea to build and run tests.

There are multiple options for building and running our tests, which you choose depends on your preferences and if you are using [CCCL's DevContainers](.devcontainer/README.md) (highly recommended!).

### Using Manual Builds Scripts
#### Building

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

#### Testing

Use the test scripts provided in the `ci/` directory to run tests for each component. These take the same arguments as the build scripts and will automatically build the tests if they haven't already been built. Running tests requires a GPU.

```bash
   ci/test_[thrust|cub|libcudacxx].sh -cxx <HOST_COMPILER> -std <CXX_STANDARD> -arch <GPU_ARCHS>
```

**Example:**
```bash
./ci/test_cub.sh -cxx g++ -std 14 -arch "70;75;80-virtual"
```

### Using CMake Presets

[CMake Presets](https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html) are a set of configurations defined in a JSON file that specify project-wide build details for CMake. They provide a standardized and sharable way to configure, build, and test projects across different platforms and development environments. Presets are available from CMake version 3.19 and later.

There are three kinds of Presets:

- Configure Presets: specify options for the `cmake` command.

- Builds Presets: specify options for the `cmake --build` command.

- Test Presets: specify options for the `ctest` command.

In CCCL we provide many presets to be used out of the box. You can find the complete list in the our corresponding [CMakePresets.json](./CMakePresets.json) file.

#### Using CMake Presets via Command Line

Once you have created your `build` directory and navigated into it, you can configure your project for a specific preset (e.g. `thrust-cpp11`) by executing the following command:

```bash
cmake --preset=thrust-cpp11 <path/to/cccl/root>
```

This command configures the project using the thrust-cpp11 preset. Please replace `<path/to/cccl/root>` with the actual path to the root directory of your CCCL project.

Upon successful configuration, initiate the build process with:

```bash
cmake --build .
```

#### Using CMake Presets via VS Code GUI extension (Recommended when using DevContainers)

The recommended way to use CMake Presets is via the VS Code extension [CMake Tools](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cmake-tools). As soon as you install the extension you would be able to see the sidebar menu below.

   ![cmaketools sidebar](/.devcontainer/img/cmaketools_sidebar.png)

You can specify the desire CMake Preset by clicking the "Select Configure Preset" button under the "Configure" drop down option (see image below).

   ![cmaketools presets](.devcontainer/img/cmaketools_presets.png)

After that you can select the default build target from the "Build" drop down option. As soon as you click the button a drop down menu will appear with all the available targets that are included within the preset you selected. For example if you had selected the `all-dev` preset VS Code will display all the available targets we have in cccl.

   ![cmaketools presets](.devcontainer/img/cmaketools_targets.png)

You can build the selected target by pressing the gear button ![gear](.devcontainer/img/build_button.png) at the bottom of the VS Code window.

Alternatively you can select the desired target from either the "Debug" or "Launch" drop down menu (for debugging or running correspondingly). <b>In that case after you select the target and either press "Run" ![run](.devcontainer/img/run.png) or "Debug" ![debug](.devcontainer/img/debug.png) the target will build on its own before running without the user having to build it explicitly from the gear button.</b>

---

We encourage users who want to debug device code to install the [Nsight Visual Studio Code Edition extension](https://marketplace.visualstudio.com/items?itemName=NVIDIA.nsight-vscode-edition) that enables the VS Code frontend for `cuda-gdb`. <u>To enable that you should avoid pressing the "Debug" button from the bottom menu and try launching it from the sidebar menu</u>.

![nsight](.devcontainer/img/nsight.png)

## Creating a Pull Request

1. Push changes to your fork
2. Create a pull request targeting the `main` branch of the original CCCL repository. Refer to [GitHub's documentation](https://docs.github.com/en/github/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests) for more information on creating a pull request.
3. Describe the purpose and context of the changes in the pull request description.

## Code Formatting (pre-commit hooks)

CCCL uses [pre-commit](https://pre-commit.com/) to execute all code linters and formatters. These
tools ensure a consistent coding style throughout the project. Using pre-commit ensures that linter
versions and options are aligned for all developers. Additionally, there is a CI check in place to
enforce that committed code follows our standards.

The linters used by CCCL are listed in `.pre-commit-config.yaml`.
For example, C++ and CUDA code is formatted with [`clang-format`](https://clang.llvm.org/docs/ClangFormat.html).

To use `pre-commit`, install via `conda` or `pip`:

```bash
conda config --add channels conda-forge
conda install pre-commit
```

```bash
pip install pre-commit
```

Then run pre-commit hooks before committing code:

```bash
pre-commit run
```

By default, pre-commit runs on staged files (only changes and additions that will be committed).
To run pre-commit checks on all files, execute:

```bash
pre-commit run --all-files
```

Optionally, you may set up the pre-commit hooks to run automatically when you make a git commit. This can be done by running:

```bash
pre-commit install
```

Now code linters and formatters will be run each time you commit changes.

You can skip these checks with `git commit --no-verify` or with the short version `git commit -n`.

## Continuous Integration (CI)

CCCL's CI pipeline tests across various CUDA versions, compilers, and GPU architectures.
For external contributors, the CI pipeline will not begin until a maintainer leaves an `/ok to test` comment. For members of the NVIDIA GitHub enterprise, the CI pipeline will begin immediately.
For a detailed overview of CCCL's CI, see [ci-overview.md](ci-overview.md).

There is a CI check for pre-commit, called [pre-commit.ci](pre-commit.ci).
This enforces that all linters (such as `clang-format`) pass.
If pre-commit.ci is failing, you can comment `pre-commit.ci autofix` on a pull request to trigger the auto-fixer.
The auto-fixer will push a commit to your pull request that applies changes made by pre-commit hooks.

## Review Process

Once submitted, maintainers will be automatically assigned to review the pull request. They might suggest changes or improvements. Constructive feedback is a part of the collaborative process, aimed at ensuring the highest quality code.

For constructive feedback and effective communication during reviews, we recommend following [Conventional Comments](https://conventionalcomments.org/).

Further recommended reading for successful PR reviews:
- [How to Do Code Reviews Like a Human (Part One)](https://mtlynch.io/human-code-reviews-1/)
- [How to Do Code Reviews Like a Human (Part Two)](https://mtlynch.io/human-code-reviews-2/)

## Thank You!

Your contributions enhance CCCL for the entire community. We appreciate your effort and collaboration!
