# This file provides utilities for building and working with cudax
# configuration targets.
#
# cudax_TARGETS
#  - Built by the calling the `cudax_build_target_list()` function.
#  - Each item is the name of a cudax interface target that is configured for a
#    certain build configuration. Currently only C++ standard dialect is
#    considered.
#
# cudax_build_target_list()
# - Creates the cudax_TARGETS list.
#
# The following functions can be used to test/set metadata on a cudax target:
#
# cudax_get_target_property(<prop_var> <target_name> <prop>)
#   - Checks the ${prop} target property on cudax target ${target_name}
#     and sets the ${prop_var} variable in the caller's scope.
#   - <prop_var> is any valid cmake identifier.
#   - <target_name> is the name of a cudax target.
#   - <prop> is one of the following:
#     - DIALECT: The C++ dialect. Valid values: 17, 20.
#     - PREFIX: A unique prefix that should be used to name all
#       targets/tests/examples that use this configuration.
#
# cudax_get_target_properties(<target_name>)
#   - Defines ${target_name}_${prop} in the caller's scope, for `prop` in:
#     {DIALECT, PREFIX}. See above for details.
#
# cudax_clone_target_properties(<dst_target> <src_target>)
#   - Set the {DIALECT, PREFIX} metadata on ${dst_target} to match
#     ${src_target}. See above for details.
#   - This *MUST* be called on any targets that link to another cudax target
#     to ensure that dialect information is updated correctly, e.g.
#     `_cn_clone_target_properties(${my_cudax_test} ${some_cudax_target})`

# Place build outputs in the root project dir:
set(cudax_LIBRARY_OUTPUT_DIR "${CMAKE_BINARY_DIR}/lib")
set(cudax_EXECUTABLE_OUTPUT_DIR "${CMAKE_BINARY_DIR}/bin")

# Define available dialects:
set(cudax_CPP_DIALECT_OPTIONS
  17 20
  CACHE INTERNAL "C++ dialects supported by CUDA Experimental." FORCE
)

# Create CMake options:
foreach (dialect IN LISTS cudax_CPP_DIALECT_OPTIONS)
  set(default_value OFF)
  if (dialect EQUAL 17) # Default to just 17 on:
    set(default_value ON)
  endif()

  option(cudax_ENABLE_DIALECT_CPP${dialect}
    "Generate C++${dialect} build configurations."
    ${default_value}
  )
endforeach()

define_property(TARGET PROPERTY _cudax_DIALECT
  BRIEF_DOCS "A target's C++ dialect."
  FULL_DOCS "A target's C++ dialect."
)
define_property(TARGET PROPERTY _cudax_PREFIX
  BRIEF_DOCS "A prefix describing the config, eg. 'cudax.cpp17'."
  FULL_DOCS "A prefix describing the config, eg. 'cudax.cpp17'."
)

function(cudax_set_target_properties target_name dialect prefix)
  set_target_properties(${target_name}
    PROPERTIES
      _cudax_DIALECT ${dialect}
      _cudax_PREFIX ${prefix}
  )

  get_target_property(type ${target_name} TYPE)
  if (NOT ${type} STREQUAL "INTERFACE_LIBRARY")
    set_target_properties(${target_name}
      PROPERTIES
        CXX_STANDARD ${dialect}
        CUDA_STANDARD ${dialect}
        ARCHIVE_OUTPUT_DIRECTORY "${cudax_LIBRARY_OUTPUT_DIR}"
        LIBRARY_OUTPUT_DIRECTORY "${cudax_LIBRARY_OUTPUT_DIR}"
        RUNTIME_OUTPUT_DIRECTORY "${cudax_EXECUTABLE_OUTPUT_DIR}"
    )
  endif()
endfunction()

# Get a cudax property from a target and store it in var_name
# _cn_get_target_property(<var_name> <target_name> [DIALECT|PREFIX]
macro(cudax_get_target_property prop_var target_name prop)
  get_property(${prop_var} TARGET ${target_name} PROPERTY _cudax_${prop})
endmacro()

# Defines the following string variables in the caller's scope:
# - ${target_name}_DIALECT
# - ${target_name}_PREFIX
macro(cudax_get_target_properties target_name)
  cudax_get_target_property(${target_name}_DIALECT ${target_name} DIALECT)
  cudax_get_target_property(${target_name}_PREFIX ${target_name} PREFIX)
endmacro()

# Set one target's _cudax_* properties to match another target
function(cudax_clone_target_properties dst_target src_target)
  cudax_get_target_properties(${src_target})
  cudax_set_target_properties(${dst_target}
    ${${src_target}_DIALECT}
    ${${src_target}_PREFIX}
  )
endfunction()

# Set ${var_name} to TRUE or FALSE in the caller's scope
function(_cn_is_config_valid var_name dialect)
  if (cudax_ENABLE_DIALECT_CPP${dialect})
    set(${var_name} TRUE PARENT_SCOPE)
  else()
    set(${var_name} FALSE PARENT_SCOPE)
  endif()
endfunction()

function(_cn_init_target_list)
  set(cudax_TARGETS "" CACHE INTERNAL "" FORCE)
endfunction()

function(_cn_add_target_to_target_list target_name dialect prefix)
  add_library(${target_name} INTERFACE)

  cudax_set_target_properties(${target_name} ${dialect} ${prefix})

  target_link_libraries(${target_name} INTERFACE
    cudax::cudax
    cudax.compiler_interface_cpp${dialect}
  )

  set(cudax_TARGETS ${cudax_TARGETS} ${target_name} CACHE INTERNAL "" FORCE)

  set(label "cpp${dialect}")
  string(TOLOWER "${label}" label)
  message(STATUS "Enabling cudax configuration: ${label}")
endfunction()

# Build a ${cudax_TARGETS} list containing target names for all
# requested configurations
function(cudax_build_target_list)
  # Clear the list of targets:
  _cn_init_target_list()

  # CMake fixed C++17 support for NVCC + MSVC targets in 3.18.3:
  if (CMAKE_CXX_COMPILER_ID STREQUAL MSVC)
    cmake_minimum_required(VERSION 3.18.3)
  endif()

  # Enable warnings in cudax headers:
  set(cudax_NO_IMPORTED_TARGETS ON)

  # Set up the cudax::cudax target while testing out our find_package scripts.
  find_package(cudax REQUIRED CONFIG
    NO_DEFAULT_PATH # Only check the explicit path in HINTS:
    HINTS "${cudax_SOURCE_DIR}"
  )

  # Build cudax_TARGETS
  foreach(dialect IN LISTS cudax_CPP_DIALECT_OPTIONS)
    _cn_is_config_valid(config_valid ${dialect})
   if (config_valid)
      set(prefix "cudax.cpp${dialect}")
      set(target_name "${prefix}")
      _cn_add_target_to_target_list(${target_name} ${dialect} ${prefix})
    endif()
  endforeach() # dialects

  list(LENGTH cudax_TARGETS count)
  message(STATUS "${count} unique cudax configurations generated")

  # Top level meta-target. Makes it easier to just build cudax targets.
  # Add all project files here so IDEs will be aware of them. This will not generate build rules.
  file(GLOB_RECURSE all_sources
    RELATIVE "${CMAKE_CURRENT_LIST_DIR}"
    "${cudax_SOURCE_DIR}/include/cuda/experimental/*.hpp"
    "${cudax_SOURCE_DIR}/include/cuda/experimental/*.cuh"
  )
  add_custom_target(cudax.all SOURCES ${all_sources})

  # Create meta targets for each config:
  foreach(cn_target IN LISTS cudax_TARGETS)
    cudax_get_target_property(config_prefix ${cn_target} PREFIX)
    add_custom_target(${config_prefix}.all)
    add_dependencies(cudax.all ${config_prefix}.all)
  endforeach()
endfunction()
