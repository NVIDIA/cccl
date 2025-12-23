# This file provides utilities for building and working with thrust
# configuration targets.
#
# THRUST_TARGETS
#  - Built by the calling the `thrust_build_target_list()` function.
#  - Each item is the name of a thrust interface target that is configured for a
#    certain combination of host/device.
#
# thrust_build_target_list()
# - Creates the THRUST_TARGETS list.
#
# The following functions can be used to test/set metadata on a thrust target:
#
# thrust_get_target_property(<prop_var> <target_name> <prop>)
#   - Checks the ${prop} target property on thrust target ${target_name}
#     and sets the ${prop_var} variable in the caller's scope.
#   - <prop_var> is any valid cmake identifier.
#   - <target_name> is the name of a thrust target.
#   - <prop> is one of the following:
#     - HOST: The host system. Valid values: CPP, OMP, TBB.
#     - DEVICE: The device system. Valid values: CUDA, CPP, OMP, TBB.
#     - PREFIX: A unique "thrust.<host>.<device>" prefix that should be used to name all
#       targets/tests/examples that use this configuration
#       (e.g. ${config_label}.test.foo).

define_property(
  TARGET
  PROPERTY _THRUST_HOST
  BRIEF_DOCS "A target's host system: CPP, TBB, or OMP."
  FULL_DOCS "A target's host system: CPP, TBB, or OMP."
)
define_property(
  TARGET
  PROPERTY _THRUST_DEVICE
  BRIEF_DOCS "A target's device system: CUDA, CPP, TBB, or OMP."
  FULL_DOCS "A target's device system: CUDA, CPP, TBB, or OMP."
)
define_property(
  TARGET
  PROPERTY _THRUST_PREFIX
  BRIEF_DOCS
    "A prefix describing the host.device config, eg. 'thrust.cpp.cuda'."
  FULL_DOCS "A prefix describing the host.device config, eg. 'thrust.cpp.cuda'."
)

function(thrust_set_target_properties target_name host device prefix)
  cccl_configure_target(${target_name})

  set_target_properties(
    ${target_name}
    PROPERTIES
      _THRUST_HOST ${host}
      _THRUST_DEVICE ${device}
      _THRUST_PREFIX ${prefix}
  )
endfunction()

# Get a thrust property from a target and store it in var_name
# thrust_get_target_property(<var_name> <target_name> [HOST|DEVICE|PREFIX]
macro(thrust_get_target_property prop_var target_name prop)
  get_property(${prop_var} TARGET ${target_name} PROPERTY _THRUST_${prop})
endmacro()

# Set ${var_name} to TRUE or FALSE in the caller's scope
function(_thrust_is_config_valid var_name host device)
  # gersemi: off
  if (THRUST_MULTICONFIG_ENABLE_SYSTEM_${host} AND
      THRUST_MULTICONFIG_ENABLE_SYSTEM_${device} AND
      "${host}_${device}" IN_LIST
        THRUST_MULTICONFIG_WORKLOAD_${THRUST_MULTICONFIG_WORKLOAD}_CONFIGS)
    # gersemi: on
    set(${var_name} TRUE PARENT_SCOPE)
  else()
    set(${var_name} FALSE PARENT_SCOPE)
  endif()
endfunction()

function(_thrust_init_target_list)
  set(THRUST_TARGETS "" CACHE INTERNAL "" FORCE)
endfunction()

function(_thrust_add_target_to_target_list target_name host device prefix)
  thrust_set_target_properties(${target_name} ${host} ${device} ${prefix})
  target_link_libraries(${target_name} INTERFACE thrust.compiler_interface)
  set(THRUST_TARGETS ${THRUST_TARGETS} ${target_name} CACHE INTERNAL "" FORCE)
  message(STATUS "Enabling Thrust configuration: ${host}.${device}")
endfunction()

function(_thrust_build_target_list_multiconfig)
  # Build THRUST_TARGETS
  foreach (host IN LISTS THRUST_HOST_SYSTEM_OPTIONS)
    foreach (device IN LISTS THRUST_DEVICE_SYSTEM_OPTIONS)
      _thrust_is_config_valid(config_valid ${host} ${device})
      if (config_valid)
        set(prefix "thrust.${host}.${device}")
        string(TOLOWER "${prefix}" prefix)

        # Configure a thrust interface target for this host/device
        set(target_name "${prefix}.config")
        thrust_create_target(
          ${target_name}
          HOST ${host}
          DEVICE ${device}
          DISPATCH ${THRUST_DISPATCH_TYPE}
          ${THRUST_TARGET_FLAGS}
        )

        # Set configuration metadata for this thrust interface target:
        _thrust_add_target_to_target_list(${target_name} ${host} ${device} ${prefix})
      endif()
    endforeach() # devices
  endforeach() # hosts

  list(LENGTH THRUST_TARGETS count)
  message(STATUS "${count} unique Thrust host.device configurations generated")
endfunction()

function(_thrust_build_target_list_singleconfig)
  set(host ${THRUST_HOST_SYSTEM})
  set(device ${THRUST_DEVICE_SYSTEM})
  set(dialect ${THRUST_CPP_DIALECT})
  set(prefix "thrust") # single config

  # Target is created in ThrustFindThrust.cmake:
  _thrust_add_target_to_target_list(thrust.config ${host} ${device} ${dialect} ${prefix})
endfunction()

# Build a ${THRUST_TARGETS} list containing target names for all
# requested configurations
function(thrust_build_target_list)
  # Clear the list of targets:
  _thrust_init_target_list()

  # Generic config flags:
  set(THRUST_TARGET_FLAGS)
  macro(add_flag_option prefix flag docstring default)
    set(opt "${prefix}_${flag}")
    option(${opt} "${docstring}" "${default}")
    mark_as_advanced(${opt})
    if ("${prefix}" STREQUAL "CCCL" AND DEFINED THRUST_${flag})
      message(
        WARNING
        "The THRUST_${flag} cmake option is deprecated. Use CCCL_${flag} instead."
      )
      set(CCCL_${flag} ${THRUST_${flag}})
    endif()
    if (${${opt}})
      list(APPEND THRUST_TARGET_FLAGS ${flag})
    endif()
  endmacro()
  # FIXME these should be moved out of the Thrust build if we care about them...
  add_flag_option(CCCL IGNORE_DEPRECATED_CPP_DIALECT "Don't warn about any deprecated C++ standards and compilers." OFF)
  add_flag_option(CCCL IGNORE_DEPRECATED_CPP_11 "Don't warn about deprecated C++11." OFF)
  add_flag_option(CCCL IGNORE_DEPRECATED_CPP_14 "Don't warn about deprecated C++14." OFF)
  add_flag_option(CCCL IGNORE_DEPRECATED_COMPILER "Don't warn about deprecated compilers." OFF)
  add_flag_option(THRUST IGNORE_CUB_VERSION_CHECK "Don't warn about mismatched CUB versions." OFF)
  add_flag_option(CCCL IGNORE_DEPRECATED_API "Don't warn about deprecated Thrust or CUB APIs." OFF)

  if (THRUST_ENABLE_MULTICONFIG)
    _thrust_build_target_list_multiconfig()
  else()
    _thrust_build_target_list_singleconfig()
  endif()
endfunction()
