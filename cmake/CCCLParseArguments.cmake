# This file defines helpers for common operations with cmake_parse_arguments.

include_guard(GLOBAL)
include(CMakeParseArguments)

# Routine error checks for argument parsing.
#
# Designed to work with cmake_parse_arguments. Assumes that:
#
# 1. The cmake_parse_arguments prefix is 'self'. Can be overridden with CPA_PREFIX.
# 2. The lists passed to cmake_parse_arguments are named:
#    - options
#    - oneValueArgs
#    - multiValueArgs
#    These can be overridden with CPA_ARG_LISTS.
#
# Usage:
#
# cmake_parse_arguments(self "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
# cccl_parse_arguments_error_checks(
#   "caller_function_name"
#   [CPA_PREFIX <cpa prefix>]
#   [CPA_ARG_LISTS <options list name> <sva list name> <mva list name>]
#   # Error if any unrecognized / unparsed arguments were given to caller:
#   [ERROR_UNPARSED]
#   # Error if any of the listed keywords were not provided to caller (empty values ok):
#   [REQUIRED_KEYWORDS kw1 [kw2]...]
#   # Error if any of the listed keywords were not provided to caller (empty values not ok):
#   [REQUIRED_VALUES kw1 [kw2]...] # Error if any of the listed keywords were not provided with non-empty values
#   # For any of the listed keywords not provided to caller, set them to the given default values:
#   [DEFAULT_VALUES kw1 val1 [kw2 val2]...]
# )
function(cccl_parse_arguments_error_checks caller_name)
  set(_tmp_options ERROR_UNPARSED)
  set(_tmp_oneValueArgs CPA_PREFIX)
  set(
    _tmp_multiValueArgs
    CPA_ARG_LISTS
    REQUIRED_KEYWORDS
    REQUIRED_VALUES
    DEFAULT_VALUES
  )
  # gersemi: hints { DEFAULT_VALUES: pairs }
  # gersemi: hints { REQUIRED_KEYWORDS: sort+unique }
  # gersemi: hints { REQUIRED_VALUES: sort+unique }
  # Parse our options:
  cmake_parse_arguments(
    __self
    "${_tmp_options}"
    "${_tmp_oneValueArgs}"
    "${_tmp_multiValueArgs}"
    ${ARGN}
  )

  # If we have unparsed arguments:
  if (__self_UNPARSED_ARGUMENTS)
    message(
      FATAL_ERROR
      "${caller_name}: Internal error: Unparsed arguments passed to cccl_parse_arguments_error_checks: ${__self_UNPARSED_ARGUMENTS}"
    )
  endif()

  set(caller_prefix self)
  if (DEFINED __self_CPA_PREFIX)
    set(caller_prefix "${__self_CPA_PREFIX}")
  endif()

  # If the caller has unparsed arguments and wants an error:
  if (__self_ERROR_UNPARSED AND DEFINED ${caller_prefix}_UNPARSED_ARGUMENTS)
    message(
      FATAL_ERROR
      "${caller_name}: Unrecognized arguments: ${${caller_prefix}_UNPARSED_ARGUMENTS}"
    )
  endif()

  # Option list names:
  set(caller_opts options)
  set(caller_svas oneValueArgs)
  set(caller_mvas multiValueArgs)
  if (DEFINED __self_CPA_ARG_LISTS)
    # Check that exactly 3 lists were provided:
    list(LENGTH __self_CPA_ARG_LISTS len)
    if (NOT len EQUAL 3)
      message(
        FATAL_ERROR
        "${caller_name}: Internal error: CPA_ARG_LISTS must specify exactly 3 list names, given ${len}."
      )
    endif()

    list(GET __self_CPA_ARG_LISTS 0 caller_opts)
    list(GET __self_CPA_ARG_LISTS 1 caller_svas)
    list(GET __self_CPA_ARG_LISTS 2 caller_mvas)
  endif()

  # Get the actual option lists:
  set(caller_opts "${${caller_opts}}")
  set(caller_svas "${${caller_svas}}")
  set(caller_mvas "${${caller_mvas}}")

  # Check that required keywords were detected, include those with empty values:
  foreach (kw IN LISTS __self_REQUIRED_KEYWORDS __self_REQUIRED_VALUES)
    set(caller_kw "${caller_prefix}_${kw}")
    # gersemi: off
    if (kw IN_LIST caller_opts AND DEFINED ${caller_kw})
      message(FATAL_ERROR "${caller_name}: Internal error: Options cannot be required keywords: '${kw}'.")
    elseif(kw IN_LIST caller_svas OR kw IN_LIST caller_mvas)
      if (DEFINED ${caller_kw} OR ${kw} IN_LIST ${caller_prefix}_KEYWORDS_MISSING_VALUES)
        continue()
      endif()
    else()
      message(FATAL_ERROR "${caller_name}: Internal error: Unrecognized required keyword: '${kw}'.")
    endif()
    message(FATAL_ERROR "${caller_name}: Required argument '${kw}' not provided.")
    # gersemi: on
  endforeach()

  # Check that required values were provided, failing for keywords with empty values:
  foreach (kw IN LISTS __self_REQUIRED_VALUES)
    set(caller_kw "${caller_prefix}_${kw}")
    # gersemi: off
    if (kw IN_LIST caller_opts)
      message(FATAL_ERROR "${caller_name}: Internal error: Options cannot have values: '${kw}'.")
    elseif(kw IN_LIST caller_svas OR kw IN_LIST caller_mvas)
      if (DEFINED ${caller_kw})
        continue()
      endif()
    else()
      message(FATAL_ERROR "${caller_name}: Internal error: Unrecognized required value keyword: '${kw}'.")
    endif()
    message(FATAL_ERROR "${caller_name}: Required argument '${kw}' not provided or missing required value.")
    # gersemi: on
  endforeach()

  # Set defaults if not defined:
  set(kw)
  foreach (iter IN LISTS __self_DEFAULT_VALUES)
    # Alternates key / values. Only single values allowed.
    if (NOT kw)
      set(kw "${iter}")
      continue()
    endif()
    set(caller_kw "${caller_prefix}_${kw}")
    set(value "${iter}")

    # gersemi: off
    if (kw IN_LIST caller_opts)
      message(FATAL_ERROR "${caller_name}: Internal error: Options cannot have default values: '${kw}'.")
    elseif (kw IN_LIST caller_svas OR kw IN_LIST caller_mvas)
      if (NOT DEFINED ${caller_kw} AND NOT ${kw} IN_LIST ${caller}_KEYWORDS_MISSING_VALUES)
        set(${caller_kw} "${value}" PARENT_SCOPE)
      endif()
    else()
      message(FATAL_ERROR "${caller_name}: Internal error: Unrecognized default value keyword: '${kw}'.")
    endif()
    set(kw) # Clear to signal next pair
    # gersemi: on
  endforeach()
endfunction()
