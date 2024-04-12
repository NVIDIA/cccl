# This file provides utilities for building and working with CudaNext
# configuration targets.
#
# CudaNext_TARGETS
#  - Built by the calling the `CudaNext_build_target_list()` function.
#  - Each item is the name of a CudaNext interface target that is configured for a
#    certain build configuration. Currently only C++ standard dialect is
#    considered.
#
# CudaNext_build_target_list()
# - Creates the CudaNext_TARGETS list.
#
# The following functions can be used to test/set metadata on a CudaNext target:
#
# CudaNext_get_target_property(<prop_var> <target_name> <prop>)
#   - Checks the ${prop} target property on CudaNext target ${target_name}
#     and sets the ${prop_var} variable in the caller's scope.
#   - <prop_var> is any valid cmake identifier.
#   - <target_name> is the name of a CudaNext target.
#   - <prop> is one of the following:
#     - DIALECT: The C++ dialect. Valid values: 17, 20.
#     - PREFIX: A unique prefix that should be used to name all
#       targets/tests/examples that use this configuration.
#
# CudaNext_get_target_properties(<target_name>)
#   - Defines ${target_name}_${prop} in the caller's scope, for `prop` in:
#     {DIALECT, PREFIX}. See above for details.
#
# CudaNext_clone_target_properties(<dst_target> <src_target>)
#   - Set the {DIALECT, PREFIX} metadata on ${dst_target} to match
#     ${src_target}. See above for details.
#   - This *MUST* be called on any targets that link to another CudaNext target
#     to ensure that dialect information is updated correctly, e.g.
#     `_cn_clone_target_properties(${my_CudaNext_test} ${some_CudaNext_target})`

# Place build outputs in the root project dir:
set(CudaNext_LIBRARY_OUTPUT_DIR "${CMAKE_BINARY_DIR}/lib")
set(CudaNext_EXECUTABLE_OUTPUT_DIR "${CMAKE_BINARY_DIR}/bin")

# Define available dialects:
set(CudaNext_CPP_DIALECT_OPTIONS
  17 20
  CACHE INTERNAL "C++ dialects supported by CudaNext." FORCE
)

# Create CMake options:
foreach (dialect IN LISTS CudaNext_CPP_DIALECT_OPTIONS)
  set(default_value OFF)
  if (dialect EQUAL 17) # Default to just 17 on:
    set(default_value ON)
  endif()

  option(CudaNext_ENABLE_DIALECT_CPP${dialect}
    "Generate C++${dialect} build configurations."
    ${default_value}
  )
endforeach()

define_property(TARGET PROPERTY _CudaNext_DIALECT
  BRIEF_DOCS "A target's C++ dialect."
  FULL_DOCS "A target's C++ dialect."
)
define_property(TARGET PROPERTY _CudaNext_PREFIX
  BRIEF_DOCS "A prefix describing the config, eg. 'cuda_next.cpp17'."
  FULL_DOCS "A prefix describing the config, eg. 'cuda_next.cpp17'."
)

function(CudaNext_set_target_properties target_name dialect prefix)
  set_target_properties(${target_name}
    PROPERTIES
      _CudaNext_DIALECT ${dialect}
      _CudaNext_PREFIX ${prefix}
  )

  get_target_property(type ${target_name} TYPE)
  if (NOT ${type} STREQUAL "INTERFACE_LIBRARY")
    set_target_properties(${target_name}
      PROPERTIES
        CXX_STANDARD ${dialect}
        CUDA_STANDARD ${dialect}
        ARCHIVE_OUTPUT_DIRECTORY "${CudaNext_LIBRARY_OUTPUT_DIR}"
        LIBRARY_OUTPUT_DIRECTORY "${CudaNext_LIBRARY_OUTPUT_DIR}"
        RUNTIME_OUTPUT_DIRECTORY "${CudaNext_EXECUTABLE_OUTPUT_DIR}"
    )
  endif()
endfunction()

# Get a CudaNext property from a target and store it in var_name
# _cn_get_target_property(<var_name> <target_name> [DIALECT|PREFIX]
macro(CudaNext_get_target_property prop_var target_name prop)
  get_property(${prop_var} TARGET ${target_name} PROPERTY _CudaNext_${prop})
endmacro()

# Defines the following string variables in the caller's scope:
# - ${target_name}_DIALECT
# - ${target_name}_PREFIX
macro(CudaNext_get_target_properties target_name)
  CudaNext_get_target_property(${target_name}_DIALECT ${target_name} DIALECT)
  CudaNext_get_target_property(${target_name}_PREFIX ${target_name} PREFIX)
endmacro()

# Set one target's _CudaNext_* properties to match another target
function(CudaNext_clone_target_properties dst_target src_target)
  CudaNext_get_target_properties(${src_target})
  CudaNext_set_target_properties(${dst_target}
    ${${src_target}_DIALECT}
    ${${src_target}_PREFIX}
  )
endfunction()

# Set ${var_name} to TRUE or FALSE in the caller's scope
function(_cn_is_config_valid var_name dialect)
  if (CudaNext_ENABLE_DIALECT_CPP${dialect})
    set(${var_name} TRUE PARENT_SCOPE)
  else()
    set(${var_name} FALSE PARENT_SCOPE)
  endif()
endfunction()

function(_cn_init_target_list)
  set(CudaNext_TARGETS "" CACHE INTERNAL "" FORCE)
endfunction()

function(_cn_add_target_to_target_list target_name dialect prefix)
  add_library(${target_name} INTERFACE)

  CudaNext_set_target_properties(${target_name} ${dialect} ${prefix})

  target_link_libraries(${target_name} INTERFACE
    CudaNext::CudaNext
    CudaNext.compiler_interface
  )

  set(CudaNext_TARGETS ${CudaNext_TARGETS} ${target_name} CACHE INTERNAL "" FORCE)

  set(label "cpp${dialect}")
  string(TOLOWER "${label}" label)
  message(STATUS "Enabling CudaNext configuration: ${label}")
endfunction()

# Build a ${CudaNext_TARGETS} list containing target names for all
# requested configurations
function(CudaNext_build_target_list)
  # Clear the list of targets:
  _cn_init_target_list()

  # CMake fixed C++17 support for NVCC + MSVC targets in 3.18.3:
  if (CMAKE_CXX_COMPILER_ID STREQUAL MSVC)
    cmake_minimum_required(VERSION 3.18.3)
  endif()

  # Enable warnings in CudaNext headers:
  set(CudaNext_NO_IMPORTED_TARGETS ON)

  # Set up the CudaNext::CudaNext target while testing out our find_package scripts.
  find_package(CudaNext REQUIRED CONFIG
    NO_DEFAULT_PATH # Only check the explicit path in HINTS:
    HINTS "${CudaNext_SOURCE_DIR}"
  )

  # Build CudaNext_TARGETS
  foreach(dialect IN LISTS CudaNext_CPP_DIALECT_OPTIONS)
    _cn_is_config_valid(config_valid ${dialect})
   if (config_valid)
      set(prefix "cuda_next.cpp${dialect}")
      set(target_name "${prefix}")
      _cn_add_target_to_target_list(${target_name} ${dialect} ${prefix})
    endif()
  endforeach() # dialects

  list(LENGTH CudaNext_TARGETS count)
  message(STATUS "${count} unique CudaNext configurations generated")

  # Top level meta-target. Makes it easier to just build CudaNext targets.
  # Add all project files here so IDEs will be aware of them. This will not generate build rules.
  file(GLOB_RECURSE all_sources
    RELATIVE "${CMAKE_CURRENT_LIST_DIR}"
    "${CudaNext_SOURCE_DIR}/include/cuda/next/*.hpp"
  )
  add_custom_target(cuda_next.all SOURCES ${all_sources})

  # Create meta targets for each config:
  foreach(cn_target IN LISTS CudaNext_TARGETS)
    CudaNext_get_target_property(config_prefix ${cn_target} PREFIX)
    add_custom_target(${config_prefix}.all)
    add_dependencies(cuda_next.all ${config_prefix}.all)
  endforeach()
endfunction()
