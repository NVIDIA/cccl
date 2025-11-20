# Adds "metatargets" using the target_name or METATARGET_PATH.
#
# A metatarget is a custom target that depends on its children targets. For example,
# a target named foo.bar.baz would create metatargets foo and foo.bar, where
# foo depends on foo.bar, and foo.bar depends on foo.bar.baz.
# This allows, for instance, `ninja cudax` to build all cudax.* targets, and `ninja cudax.test`
# to build all cudax.test.* targets.
function(cccl_ensure_metatargets target_name)
  set(options)
  set(oneValueArgs METATARGET_PATH)
  set(multiValueArgs)
  cmake_parse_arguments(
    _cccl
    "${options}"
    "${oneValueArgs}"
    "${multiValueArgs}"
    ${ARGN}
  )

  if (_cccl_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "Unrecognized arguments: ${_cccl_UNPARSED_ARGUMENTS}")
  endif()

  if (NOT DEFINED _cccl_METATARGET_PATH)
    set(_cccl_METATARGET_PATH ${target_name})
  endif()

  set(parent_path "")
  set(current_path "")
  string(REPLACE "." ";" path_parts "${_cccl_METATARGET_PATH}")
  foreach (part IN LISTS path_parts)
    if (current_path STREQUAL "")
      set(current_path "${part}")
    else()
      set(current_path "${current_path}.${part}")
    endif()

    if (NOT TARGET ${current_path})
      add_custom_target(${current_path})
    endif()

    if (NOT parent_path STREQUAL "")
      add_dependencies(${parent_path} ${current_path})
    endif()

    set(parent_path ${current_path})
  endforeach()

  if (NOT target_name STREQUAL current_path)
    add_dependencies(${current_path} ${target_name})
  endif()
endfunction()
