# Test the CMake packages for CCCL and all subprojects.
#
# Parameters:
# - CCCL_ROOT [Path] Root of the CCCL repo, or an installation root.
# - COMPONENTS [StringList] {Thrust CUB libcudacxx} Which CCCL subprojects
#   should be found.
# - PACKAGE_TYPE [String] {CCCL | NATIVE | SUBDIR}:
#   - CCCL -> `find_package(CCCL COMPONENTS <subproject>)`
#   - NATIVE -> `find_package(<subproject>)`
#   - SUBDIR -> `set(CCCL_REQUIRED_COMPONENTS <subproject>)`
#               `add_subdirectory(${cccl_root})`

cmake_minimum_required(VERSION 3.21)
project(CCCLTestExport LANGUAGES CXX)

include(CTest)
enable_testing()

set(CCCL_ROOT "" CACHE PATH "Root of the CCCL repo, or an installation root.")
set(
  COMPONENTS
  ""
  CACHE STRING
  "DEFAULT for no components, or semi-colon delimited list of Thrust, CUB, and/or libcudacxx."
)
set(
  PACKAGE_TYPE
  ""
  CACHE STRING
  "CCCL: Find CCCL with subpackages as components; NATIVE: Find subpackages directly; SUBDIR: add_subdirectory(${CCCL_ROOT}"
)
set_property(CACHE PACKAGE_TYPE PROPERTY STRINGS CCCL NATIVE SUBDIR)

message(STATUS "CCCL_ROOT=${CCCL_ROOT}")
message(STATUS "COMPONENTS=${COMPONENTS}")
message(STATUS "PACKAGE_TYPE=${PACKAGE_TYPE}")

function(do_find_package pkg_name pkg_prefix)
  list(
    APPEND arg_list
    REQUIRED
    ${ARGN}
    NO_DEFAULT_PATH
    HINTS
    "${pkg_prefix}"
  )
  list(JOIN arg_list " " arg_str)
  message(STATUS "Executing: find_package(${pkg_name} ${arg_str})")
  find_package(${pkg_name} ${arg_list})
  if (NOT ${pkg_name}_FOUND)
    message(FATAL_ERROR "Failed: find_package(${pkg_name} ${arg_str})")
  endif()
  # Re-execute find_package to ensure that repeated calls don't break:
  find_package(${pkg_name} ${arg_list})
endfunction()

# Run find package with the requested configuration:
if (PACKAGE_TYPE STREQUAL "CCCL")
  if (COMPONENTS STREQUAL "DEFAULT")
    do_find_package(CCCL "${CCCL_ROOT}")
  else()
    do_find_package(CCCL "${CCCL_ROOT}" COMPONENTS ${COMPONENTS})
  endif()
elseif (PACKAGE_TYPE STREQUAL "NATIVE")
  if (COMPONENTS STREQUAL "DEFAULT")
    message(
      FATAL_ERROR
      "COMPONENTS=DEFAULT incompatible with PACKAGE_TYPE=NATIVE"
    )
  endif()
  foreach (component IN LISTS COMPONENTS)
    do_find_package(${component} "${CCCL_ROOT}")
  endforeach()
elseif (PACKAGE_TYPE STREQUAL "SUBDIR")
  if (COMPONENTS STREQUAL "DEFAULT")
    set(CCCL_REQUIRED_COMPONENTS)
  else()
    set(CCCL_REQUIRED_COMPONENTS ${COMPONENTS})
  endif()
  add_subdirectory("${CCCL_ROOT}" "${CMAKE_CURRENT_BINARY_DIR}/subdir")
else()
  message(FATAL_ERROR "Invalid PACKAGE_TYPE: ${PACKAGE_TYPE}")
endif()

if (COMPONENTS STREQUAL "DEFAULT")
  set(COMPONENTS libcudacxx CUB Thrust)
  if (CCCL_ENABLE_UNSTABLE)
    list(APPEND COMPONENTS cudax)
  endif()
endif()

foreach (component IN LISTS COMPONENTS)
  set(test_target version_check.${component})
  set(component_target "${component}::${component}")
  add_executable(${test_target} version_check.cxx)
  target_compile_features(${test_target} PUBLIC cxx_std_17)
  target_link_libraries(${test_target} PRIVATE ${component_target})
  add_test(NAME ${test_target} COMMAND ${test_target})

  if (component STREQUAL "libcudacxx")
    math(
      EXPR
      component_cmake_version
      "(${LIBCUDACXX_VERSION_MAJOR} * 1000000) +
        ${LIBCUDACXX_VERSION_MINOR} * 1000 +
        ${LIBCUDACXX_VERSION_PATCH}"
    )
    target_compile_definitions(
      ${test_target}
      PRIVATE
        "VERSION_HEADER=cuda/std/version"
        "VERSION_MACRO=_LIBCUDACXX_CUDA_API_VERSION"
        "EXPECTED_VERSION=${component_cmake_version}"
    )
  elseif (component STREQUAL "CUB")
    math(
      EXPR
      component_cmake_version
      "(${CUB_VERSION_MAJOR} * 100000) +
        ${CUB_VERSION_MINOR} * 100 +
        ${CUB_VERSION_PATCH}"
    )
    target_compile_definitions(
      ${test_target}
      PRIVATE
        "VERSION_HEADER=cub/version.cuh"
        "VERSION_MACRO=CUB_VERSION"
        "EXPECTED_VERSION=${component_cmake_version}"
    )
  elseif (component STREQUAL "Thrust")
    math(
      EXPR
      component_cmake_version
      "(${THRUST_VERSION_MAJOR} * 100000) +
        ${THRUST_VERSION_MINOR} * 100 +
        ${THRUST_VERSION_PATCH}"
    )
    target_compile_definitions(
      ${test_target}
      PRIVATE
        "VERSION_HEADER=thrust/version.h"
        "VERSION_MACRO=THRUST_VERSION"
        "EXPECTED_VERSION=${component_cmake_version}"
    )
  elseif (component STREQUAL "cudax")
    get_target_property(cudax_aliased_target cudax::cudax ALIASED_TARGET)
    if (NOT cudax_aliased_target STREQUAL "_cudax_cudax")
      message(
        FATAL_ERROR
        "Expected cudax::cudax to alias the non-imported _cudax_cudax target; got '${cudax_aliased_target}'"
      )
    endif()

    get_target_property(cudax_target_imported _cudax_cudax IMPORTED)
    if (cudax_target_imported)
      message(FATAL_ERROR "Expected _cudax_cudax to be a non-imported target")
    endif()

    get_target_property(
      cudax_compile_features
      _cudax_cudax
      INTERFACE_COMPILE_FEATURES
    )
    foreach (required_feature IN ITEMS cxx_std_17 cuda_std_17)
      if (NOT required_feature IN_LIST cudax_compile_features)
        message(
          FATAL_ERROR
          "Expected cudax::cudax to require ${required_feature}; got '${cudax_compile_features}'"
        )
      endif()
    endforeach()

    get_target_property(
      cudax_compile_definitions
      _cudax_cudax
      INTERFACE_COMPILE_DEFINITIONS
    )
    if (
      NOT
        _CUDAX_ENABLE_GROUP_FEATURES_IN_LIBCUDACXX
          IN_LIST
          cudax_compile_definitions
    )
      message(
        FATAL_ERROR
        "Expected cudax::cudax to enable libcu++ group features; got '${cudax_compile_definitions}'"
      )
    endif()

    math(
      EXPR
      component_cmake_version
      "(${CUDAX_VERSION_MAJOR} * 1000000) +
        ${CUDAX_VERSION_MINOR} * 1000 +
        ${CUDAX_VERSION_PATCH}"
    )
    target_compile_definitions(
      ${test_target}
      PRIVATE
        "VERSION_HEADER=cuda/experimental/version.cuh"
        "VERSION_MACRO=CUDAX_VERSION"
        "EXPECTED_VERSION=${component_cmake_version}"
    )
  else()
    message(
      FATAL_ERROR
      "Valid COMPONENTS are (case-sensitive): Thrust;CUB;libcudacxx;cudax"
    )
  endif()
endforeach()
