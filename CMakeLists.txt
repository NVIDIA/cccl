# 3.18 is the minimum for including the project with add_subdirectory.
# 3.21 is the minimum for the developer build.
cmake_minimum_required(VERSION 3.18)

# sccache cannot handle the -Fd option generating pdb files:
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
endif()

# Determine whether CCCL is the top-level project or included into
# another project via add_subdirectory()

if (NOT DEFINED CCCL_TOPLEVEL_PROJECT)
  # DO NOT REMOVE THE FOLLOWING LINE
  set(CCCL_TOPLEVEL_PROJECT OFF)
  # REQUIRED FOR CCCL TO WORK VIA CPM WITH INSTALL RULES

  if ("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_LIST_DIR}")
    set(CCCL_TOPLEVEL_PROJECT ON)
  endif()
endif()

# Enable CXX so CMake can configure install paths
project(CCCL LANGUAGES CXX)

include(cmake/CCCLInstallRules.cmake)

# Support adding CCCL to a parent project via add_subdirectory.
include(cmake/CCCLAddSubdirHelper.cmake) # Always include, this is used by subprojects as well.
if (NOT CCCL_TOPLEVEL_PROJECT)
  include(cmake/CCCLAddSubdir.cmake)
endif()

if (CCCL_TOPLEVEL_PROJECT)
  # We require a higher cmake version for dev builds
  cmake_minimum_required(VERSION 3.21)

  # Handle special CCCL values for CMAKE_CUDA_ARCHITECTURES
  include(cmake/CCCLCheckCudaArchitectures.cmake)
  cccl_check_cuda_architectures()

  # Perform developer build checks
  include(cmake/CCCLDevBuildChecks.cmake)
  cccl_dev_build_checks()
endif()

option(CCCL_ENABLE_LIBCUDACXX "Enable the libcu++ developer build." OFF)
option(CCCL_ENABLE_CUB "Enable the CUB developer build." OFF)
option(CCCL_ENABLE_THRUST "Enable the Thrust developer build." OFF)
option(CCCL_ENABLE_TESTING "Enable CUDA C++ Core Library tests." OFF)
option(CCCL_ENABLE_EXAMPLES "Enable CUDA C++ Core Library examples." OFF)
option(CCCL_ENABLE_C_PARALLEL "Enable CUDA C Parallel Library." OFF)
option(CCCL_ENABLE_C_EXPERIMENTAL_STF "Enable CUDA C CUDASTF Library." OFF)
option(CCCL_ENABLE_NVBENCH_HELPER "Enable the NVBench Helper Dev Build." OFF)

if ("NVHPC" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
  set(CCCL_ENABLE_BENCHMARKS OFF)
else()
  option(CCCL_ENABLE_BENCHMARKS "Enable CUDA C++ Core Library benchmarks." OFF)
endif()

option(
  CCCL_ENABLE_UNSTABLE
  "Enable targets and developer build options for unstable projects."
  OFF
)
if (CCCL_ENABLE_UNSTABLE)
  option(
    CCCL_ENABLE_CUDAX
    "Enable the CUDA Experimental developer build."
    ${CCCL_TOPLEVEL_PROJECT}
  )
else()
  # Always off if unstable disabled:
  # Note that this doesn't override the cache variable, but rather creates a new
  # directory-scoped variable that shadows it. This is sufficient for our purposes.
  set(CCCL_ENABLE_CUDAX OFF)
endif()

include(CTest)
enable_testing()

option(CCCL_USE_LIBCXX "Use libc++ when compiling and linking" OFF)

if (CCCL_TOPLEVEL_PROJECT)
  include(cmake/AppendOptionIfAvailable.cmake)
  include(cmake/CCCLUtilities.cmake) # include this before other CCCL helpers

  include(cmake/CCCLAddExecutable.cmake)
  include(cmake/CCCLBuildCompilerTargets.cmake)
  include(cmake/CCCLClangdCompileInfo.cmake)
  include(cmake/CCCLConfigureTarget.cmake)
  include(cmake/CCCLEnsureMetaTargets.cmake)
  include(cmake/CCCLGenerateHeaderTests.cmake)
  include(cmake/CCCLGetDependencies.cmake)
  include(cmake/CCCLTestParams.cmake)

  cccl_build_compiler_targets()
endif()

add_subdirectory(libcudacxx)
add_subdirectory(cub)
add_subdirectory(thrust)

if (CCCL_ENABLE_UNSTABLE)
  add_subdirectory(cudax)
endif()

if (CCCL_ENABLE_C_PARALLEL OR CCCL_ENABLE_C_EXPERIMENTAL_STF)
  add_subdirectory(c)
endif()

if (CCCL_ENABLE_TESTING)
  add_subdirectory("ci")
  add_subdirectory("test")
endif()

if (CCCL_ENABLE_EXAMPLES)
  add_subdirectory(examples)
endif()

if (CCCL_ENABLE_NVBENCH_HELPER)
  add_subdirectory(nvbench_helper)
endif()

# Must stay at the end of this file.
include(cmake/CCCLHideThirdPartyOptions.cmake)
