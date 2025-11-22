# This file defines helpers for common operations with cmake_parse_arguments.

include_guard(GLOBAL)
include(CMakeParseArguments)

# Routine error checks for argument parsing.
#
# Required positional arguments:
# caller_name: Name of the calling function (for error messages).
# caller_prefix: Prefix used for the caller's cmake_parse_arguments.
# caller_opts: List of option keywords used by the caller.
# caller_svas: List of single-value argument keywords used by the caller.
# caller_mvas: List of multi-value argument keywords used by the caller.
#
# Checks:
# [ERROR_UNPARSED]
#   Checks that there were no unparsed arguments.
# [REQUIRED_KEYWORDS kw1 [kw2]...]
#   Check that the listed keywords were provided (even if with empty values).
# [REQUIRED_VALUES kw1 [kw2]...]
#   Check that the listed keywords were provided with non-empty values.
# [DEFAULT_VALUES kw1 val1 [kw2 val2]...]
#   For any of the listed keywords not provided, set them to the given default
#   values.
function(
  cccl_parse_arguments_error_checks
  caller_name
  caller_prefix
  caller_opts
  caller_svas
  caller_mvas
)
  set(options ERROR_UNPARSED)
  set(singleValueArgs "")
  set(multiValueArgs REQUIRED_KEYWORDS REQUIRED_VALUES DEFAULT_VALUES)
  # gersemi: hints { DEFAULT_VALUES: pairs }
  # gersemi: hints { REQUIRED_KEYWORDS: sort+unique }
  # gersemi: hints { REQUIRED_VALUES: sort+unique }
  cmake_parse_arguments(
    __self
    "${options}"
    "${singleValueArgs}"
    "${multiValueArgs}"
    ${ARGN}
  )

  if (__self_ERROR_UNPARSED AND DEFINED ${caller_prefix}_UNPARSED_ARGUMENTS)
    message(
      FATAL_ERROR
      "${caller_name}: Unrecognized arguments: ${${caller_prefix}_UNPARSED_ARGUMENTS}"
    )
  endif()

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
