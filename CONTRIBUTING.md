
# Contributing to CCCL

First and foremost, thank you for considering contributing to CCCL! Your efforts and expertise will help improve this project for everyone.

This document provides guidelines and instructions for contributing to the project. By adhering to these guidelines, you ensure that the contribution process is efficient and effective for everyone involved.

## Table of Contents

- [Setting Up Your Development Environment](#setting-up-your-development-environment)
- [Making Changes](#making-changes)
- [Submitting a Pull Request](#submitting-a-pull-request)
- [Coding Standards](#coding-standards)
- [Code of Conduct](#code-of-conduct)

## Setting Up Your Development Environment

1. **Fork the Repository**: Start by forking the CCCL repository to your own GitHub account.
2. **Clone Your Fork**: `git clone https://github.com/YOUR_USERNAME/cccl.git`
3. **Set Up Devcontainer**: If you're using Visual Studio Code, open the project and launch the desired `.devcontainer` environment. This ensures you have all the necessary dependencies and tools.
4. **Install Additional Dependencies**: (If any specific to the project, list them here.)

## Making Changes

1. **Create a New Branch**: Always create a new branch for your changes: `git checkout -b feature/your-feature-name` or `git checkout -b fix/your-fix-name`.
2. **Make Your Changes**: Develop your feature or fix, adhering to the coding standards mentioned below.
3. **Commit Your Changes**: Make sure to write clear and concise commit messages.
4. **Push to Your Fork**: `git push origin feature/your-feature-name`.

## Build Tests

1. **Set Up the Environment**: Ensure you're in the correct directory, which is the directory containing the build script you intend to run.
2. **Choose the Component**: Depending on which component of CCCL you're working on (`cub`, `libcudacxx`, or `thrust`), you will use the respective build script.
3. **Execute the Build Script**: The general form for executing a build script is:
   ```bash
   ci/build_component.sh [OPTIONS] <HOST_COMPILER> <CXX_STANDARD> <GPU_ARCHS>
   ```
   For example, to build `libcudacxx` with `g++-8`, C++14 standard, and GPU architecture 70, you'd run:
   ```bash
   ./ci/build_libcudacxx.sh g++-8 14 "70"
   ```

Options:
- `-nvcc`: Specify a custom path to `nvcc` if not using the default.
- `-v/--verbose`: Enable verbose mode for debugging.

Remember to ensure all components build successfully with your modifications before proceeding.

## Run Tests

1. **Ensure the Project is Built**: Before running tests for `cub` and `thrust`, make sure the respective component is built using the appropriate build script.
2. **Choose the Component**: Depending on which component of CCCL you're working on (`cub`, `libcudacxx`, or `thrust`), you will use the respective test script.
3. **Execute the Test Script**: Run the desired test script, e.g., to test `libcudacxx`:
   ```bash
   ./ci/test_libcudacxx.sh
   ```
4. **Inspect the Results**: After the tests run, review the results to ensure all tests pass. If any tests fail, investigate and address the issues before submitting your changes.

## Submitting a Pull Request

1. **Update Your Fork**: Ensure your fork is up-to-date with the main CCCL repository.
2. **Initiate a Pull Request**: From your fork, click on the 'New Pull Request' button.
3. **Describe Your Changes**: In the pull request description, explain the changes you made, the issues they resolve, and any other relevant context.
4. **Wait for a Review**: Maintainers will review your pull request, provide feedback, or request changes if necessary. Address any feedback to get your changes merged.

## Coding Standards

- **Code Formatting**: Ensure your code adheres to the formatting standards used throughout the project. (Mention any specific linters or formatters if used.)
- **Comments**: Comment your code where necessary to explain complex or non-intuitive sections.
- **Tests**: If your changes introduce new functionality, make sure to include appropriate tests.
- **Documentation**: Update the documentation to reflect your changes, if applicable.

## Code of Conduct

All contributors are expected to adhere to the project's [Code of Conduct](CODE_OF_CONDUCT.md). Please ensure you read and understand its contents.
