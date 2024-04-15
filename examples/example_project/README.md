
# Example Project Using CCCL From GitHub

Many CUDA C++ users are accustomed to using CCCL headers (Thrust, CUB, libcu++) provided with the [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) or [NVIDIA HPC SDK](https://developer.nvidia.com/hpc-sdk).

In addition, we also support using CCCL directly from GitHub.
The primary benefit is that this allows users to use the latest version of CCCL without having to wait for a new release of the CUDA Toolkit or HPC SDK.

This example demonstrates how to use CCCL from GitHub in a CMake project.

## Overview

This is a standalone example of how to use [CCCL](https://github.com/nvidia/cccl) in a CMake project.

This example demonstrates fetching CCCL from GitHub and linking it with a simple example CUDA program ([`example.cu`](example.cu)) that utilizes the headers from CCCL.

This is intended to be a starting point for users who want to use CCCL in their own projects.

## How to Adapt This Example to Your Project

This example is intended to be a starting point for users who want to use CCCL in their own projects.
In order to adapt this example to your project, you will need to do the following:
1. Download `CPM.cmake` into your project's `cmake/` directory ([see below for instructions](#downloading-cpm)).
2. Add the following lines to your project's `CMakeLists.txt` file:
   ```cmake
   include(cmake/CPM.cmake)

   # This will automatically clone CCCL from GitHub and make the exported cmake targets available
   CPMAddPackage(
       NAME CCCL
       GITHUB_REPOSITORY nvidia/cccl
       GIT_TAG main # Fetches the latest commit on the main branch
   )

   # If you're building an executable
   add_executable(your_executable your_file.cu)
   target_link_libraries(your_executable PRIVATE CCCL::CCCL)

   # Alternatively, if you're building a library
   add_library(your_library your_file.cu)
   target_link_libraries(your_library PRIVATE CCCL::CCCL)
   ```
   See the [CMakeLists.txt](CMakeLists.txt) file in this directory for a complete example.
3. Configure and build your project as normal and verify that it builds successfully.

For more information on using CPM, see [below](#using-cmake-package-manager).

## Using CMake Package Manager

This example uses the CMake Package Manager (CPM) to fetch CCCL from GitHub.

See the [CMakeLists.txt](CMakeLists.txt) file in this directory for the complete example.

If you are not familiar with CPM, you can find more information [here](https://github.com/cpm-cmake/CPM.cmake).
In short, CPM is a CMake module that simplifies dependency management for CMake projects.
It automatically downloads and integrates dependencies into your CMake project.

### Downloading CPM

In order to get the latest version of CPM.cmake, you can run the following command in the root directory of your project:

```bash
mkdir -p cmake
wget -O cmake/CPM.cmake https://github.com/cpm-cmake/CPM.cmake/releases/latest/download/get_cpm.cmake
```

This will download and create the file `cmake/CPM.cmake` in your project directory.
Most projects will want to commit this file to their source control system.
You can then use `include(cmake/CPM.cmake)` in your project's `CMakeLists.txt` file to include CPM in your project.

Alternatively, you can add the following logic to your `CMakeLists.txt` to download CPM if it is not already present in your project directory.

```cmake
set(CPM_DOWNLOAD_VERSION 0.34.0)

if(CPM_SOURCE_CACHE)
  set(CPM_DOWNLOAD_LOCATION "${CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
elseif(DEFINED ENV{CPM_SOURCE_CACHE})
  set(CPM_DOWNLOAD_LOCATION "$ENV{CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
else()
  set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
endif()

if(NOT (EXISTS ${CPM_DOWNLOAD_LOCATION}))
  message(STATUS "Downloading CPM.cmake to ${CPM_DOWNLOAD_LOCATION}")
  file(DOWNLOAD
       https://github.com/TheLartians/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake
       ${CPM_DOWNLOAD_LOCATION}
  )
endif()

include(${CPM_DOWNLOAD_LOCATION})
```

## Building and Running the Example

Most people will want to adapt this example to their own project as described [above](#how-to-adapt-this-example-to-your-project). If you would like to build and run this example as-is, you will need to follow the instructions below.

### Prerequisites

If you would like to build and run this example as-is, you will need:

- A CUDA-capable GPU
- NVIDIA CUDA Toolkit (11.1 or later)
- CMake (3.14 or later)
- A C++14 standard-compliant compiler
- git

### Instructions

1. Clone this repository to your local machine.
   ```bash
   git clone https://github.com/NVIDIA/cccl.git
   ```

2. Enter the directory of the cloned repository.
   ```bash
   cd cccl/examples/example_project
   ```

3. Run the CMake configure step
   ```bash
   cmake -S . -B build
   ```
   Alternatively,
   ```bash
   mkdir -p build
   cd build
   cmake ..
   ```
4. Run the CMake build step.
   ```bash
   cmake --build .
   ```

6. Run the executable.
   ```bash
   ./build/example_project
   ```

If everything is configured correctly, the program will execute and print the sum of an array of integers, demonstrating the use of cccl.
