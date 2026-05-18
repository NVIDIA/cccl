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

# Enable CCCL specific platform files
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

include(cmake/CCCLInstallRules.cmake)

# Support adding CCCL to a parent project via add_subdirectory.
include(cmake/CCCLAddSubdirHelper.cmake) # Always include, this is used by subprojects as well.
if (NOT CCCL_TOPLEVEL_PROJECT)
  include(cmake/CCCLAddSubdir.cmake)
endif()

if (CCCL_TOPLEVEL_PROJECT AND NOT CCCL_SKIP_BUILD_CHECKS)
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

option(CCCL_ENABLE_TILE "Enable tile support" OFF)

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

option(CCCL_ENABLE_CLANG_TIDY "Enable clang-tidy" OFF)
mark_as_advanced(CCCL_ENABLE_CLANG_TIDY)

if (CCCL_TOPLEVEL_PROJECT)
  include(cmake/CCCLAddTidyTarget.cmake)

  if (CCCL_ENABLE_CLANG_TIDY)
    cccl_tidy_init()
  endif()
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

# CUDA runtime smoke test.
#
# A single CTest fixture-setup test verifies that the CUDA runtime is functional
# (device visible, kernel launch + managed-memory round-trip work). Every other
# CCCL test in this build is then declared to require that fixture, so a broken
# CUDA install fails fast and once, instead of producing dozens of confusing
# downstream failures.
#
# We deliberately gate on the top-level CCCL_ENABLE_* options here, not on the
# subprojects' *_ENABLE_TESTING options: those are option()s declared inside
# each subproject's CMakeLists.txt and are therefore undefined at this scope
# until add_subdirectory() runs. Note also that the smoke test is intentionally
# *not* gated on CCCL_ENABLE_TESTING: it is meant to guard any GPU-touching
# test that any enabled subproject registers, regardless of whether the
# top-level test/ tree is also being built.
set(CCCL_CUDA_SMOKE_TARGET cccl.test.cuda_runtime_smoke)
set(CCCL_CUDA_SMOKE_FIXTURE cccl.cuda_runtime_ok)

set(CCCL_CUDA_SMOKE_ENABLED OFF)
if (
  CCCL_TOPLEVEL_PROJECT
  AND
    (
      CCCL_ENABLE_CUB
      OR CCCL_ENABLE_THRUST
      OR CCCL_ENABLE_LIBCUDACXX
      OR CCCL_ENABLE_CUDAX
    )
)
  set(CCCL_CUDA_SMOKE_ENABLED ON)
endif()

# cmake/CCCLCudaSmokeFixture.cmake shadows add_test() with a macro that
# auto-attaches FIXTURES_REQUIRED=${CCCL_CUDA_SMOKE_FIXTURE} to every CUDA-
# touching test. Must be included BEFORE any add_subdirectory() that may
# register tests, otherwise the macro shadow misses early tests.
if (CCCL_TOPLEVEL_PROJECT)
  include(cmake/CCCLCudaSmokeFixture.cmake)
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

# Must run after every subproject has registered its tests. Fixture properties
# are attached from each subproject's CMakeLists.txt (see
# cmake/CCCLCudaSmokeFixture.cmake) so set_tests_properties sees nested-project
# tests; then register the smoke target here.
if (CCCL_CUDA_SMOKE_ENABLED)
  # test/cuda_smoke/ declares its own project(... LANGUAGES CXX CUDA), so the
  # CUDA toolchain is enabled inside that subtree only. The root project()
  # above stays CXX-only; external consumers that pull CCCL in via
  # add_subdirectory() see no change in root-scope language enablement.
  add_subdirectory(test/cuda_smoke)

  # Make each subproject's top-level metatarget pull in the smoke binary.
  # This means `ninja cub`/`thrust`/`libcudacxx`/`cudax` (which most test
  # presets build, directly or transitively) always produces the smoke
  # executable, so no CMakePresets.json target list needs to mention it by
  # name. Presets that build a more specific target still need to either list
  # the smoke target or build one of these metatargets.
  foreach (_sub IN ITEMS libcudacxx cub thrust cudax)
    if (TARGET ${_sub})
      add_dependencies(${_sub} ${CCCL_CUDA_SMOKE_TARGET})
    endif()
  endforeach()
endif()

if (CCCL_ENABLE_EXAMPLES)
  add_subdirectory(examples)
endif()

if (CCCL_ENABLE_NVBENCH_HELPER)
  add_subdirectory(nvbench_helper)
endif()

# Must stay at the end of this file.
include(cmake/CCCLHideThirdPartyOptions.cmake)
