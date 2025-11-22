include(CMakeParseArguments)

# A nicer wrapper around cmake_parse_arguments:
#
# cccl_parse_arguments(
#   <parent_name>                        # Required; function / macro calling this helper. Must be first.
#   [OPTIONS <option1> <option2> ...]    # Optional; If specified, passed to cmake_parse_arguments.
#   [ONE_VALUE_ARGS <arg1> <arg2> ...]   # Optional; If specified, passed to cmake_parse_arguments.
#   [MULTI_VALUE_ARGS <arg1> <arg2> ...] # Optional; If specified, passed to cmake_parse_arguments.
#   [ERROR_IF_UNPARSED]                  # Optional; If specified, unparsed args cause fatal error.
#   [PREFIX <prefix>]                    # Optional; default: "self"
#   [PARENT_NAME <parent_scope_name>]    # Optional; function / macro for nicer error messages.
#   PARENT_ARGN <parent_argn>            # Required, must be last.
# )

function(cccl_parse_arguments)
  # Provided def in .gersemi/ext/cccl.py:
  # gersemi: ignore
  set(options ERROR_IF_UNPARSED)
  set(oneValueArgs PREFIX PARENT_NAME)
  set(multiValueArgs PARENT_ARGN OPTIONS ONE_VALUE_ARGS MULTI_VALUE_ARGS)
  cmake_parse_arguments(
    "__self_"
    "${options}"
    "${oneValueArgs}"
    "${multiValueArgs}"
    ${ARGN}
  )

  if (NOT DEFINED "__self_PARENT_ARGN")
    message(FATAL_ERROR "cccl_parse_arguments requires PARENT_ARGN to be defined.")
  endif()

  if (NOT DEFINED "__self_PREFIX")
    set(__self_PREFIX "self")
  endif()

 if (NOT DEFINED "__self_PARENT_NAME")
   set(__self_PARENT_NAME "cccl_parse_arguments's parent function")
 endif()

  # Reset args for cmake_parse_arguments call:
  set(options)
  if (DEFINED "__self_OPTIONS")
    set(options "${__self_OPTIONS}")
  endif()

  set(one_value_args)
  if (DEFINED "__self_ONE_VALUE_ARGS")
    set(one_value_args "${__self_ONE_VALUE_ARGS}")
  endif()

  set(multi_value_args)
  if (DEFINED "__self_MULTI_VALUE_ARGS")
    set(multi_value_args "${__self_MULTI_VALUE_ARGS}")
  endif()

  cmake_parse_arguments(
    "${__self_PREFIX}"
    "${options}"
    "${one_value_args}"
    "${multi_value_args}"
    ${__self_PARENT_ARGN}
  )


  if (DEFINED "${__self_PREFIX}_UNPARSED_ARGUMENTS")
    if (DEFINED "__self_NO_UNPARSED")
      message(FATAL_ERROR "${${__self_PREFIX}_PARENT_NAME} given invalid arguments: ${${__self_PREFIX}_UNPARSED_ARGUMENTS}")
    else()
      set("${__self_PREFIX}_UNPARSED_ARGUMENTS" "${${__self_PREFIX}_UNPARSED_ARGUMENTS}" PARENT_SCOPE)
    endif()
  else()
    unset("${__self_PREFIX}_UNPARSED_ARGUMENTS" PARENT_SCOPE)
  endif()

  foreach(arg IN LISTS options one_value_args multi_value_args)
    if (DEFINED "${__self_PREFIX}_${arg}")
      set("${__self_PREFIX}_${arg}" "${${__self_PREFIX}_${arg}}" PARENT_SCOPE)
    else()
      unset("${__self_PREFIX}_${arg}" PARENT_SCOPE)
    endif()
  endforeach()
endfunction()
