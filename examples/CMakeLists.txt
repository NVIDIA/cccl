# Set default THRUST_BACKEND if not provided
if (NOT DEFINED THRUST_BACKEND)
  set(THRUST_BACKEND "CUDA")
endif()

# Escape semicolons in architectures
string(REPLACE ";" "\\\;" arches_escaped "${CMAKE_CUDA_ARCHITECTURES}")

# CMake options for building the example
set(cmake_opts
  -D "CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
  -D "CMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}"
  -D "CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
  -D "CMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}"
  -D "CMAKE_CUDA_HOST_COMPILER=${CMAKE_CUDA_HOST_COMPILER}"
  -D "CMAKE_CUDA_ARCHITECTURES=${arches_escaped}"
  -D "THRUST_BACKEND=${THRUST_BACKEND}"
)

set(CCCL_EXAMPLE_CPM_REPOSITORY "nvidia/cccl" CACHE STRING "GitHub repository used for CPM examples.")
set(CCCL_EXAMPLE_CPM_TAG "main" CACHE STRING "Git tag/branch used for CPM examples.")

set(cmake_cpm_opts
  -D "CCCL_REPOSITORY=${CCCL_EXAMPLE_CPM_REPOSITORY}"
  -D "CCCL_TAG=${CCCL_EXAMPLE_CPM_TAG}"
)

# Add the device_system_selector example
cccl_add_compile_test(test_name
  cccl.example
  device_system_selector
  "default"
  ${cmake_opts}
  ${cmake_cpm_opts}
)

find_package(CUDAToolkit REQUIRED)

if (CUDAToolkit_VERSION_MAJOR VERSION_GREATER_EQUAL 12)
  cccl_add_compile_test(test_name
    cccl.example
    cudax
    "default"
    ${cmake_opts}
    ${cmake_cpm_opts}
  )
endif()
