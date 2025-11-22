# Adds an executable target from SOURCES with standard CCCL configuration.
#
# By default, metatargets are created (e.g. target name foo.bar.baz will built by metatargets
# `foo` and `foo.bar`). This can be disabled with NO_METATARGETS. By default, the metatarget
# path is the same as the target name, but can be overridden with METATARGET_PATH
#
# If ADD_CTEST is specified, a CTest test is added with the same name as the target,
# which runs the executable with no arguments.
function(cccl_add_executable target_name)
  set(options ADD_CTEST NO_METATARGETS)
  set(oneValueArgs METATARGET_PATH)
  set(multiValueArgs SOURCES)
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

  if (NOT DEFINED _cccl_SOURCES)
    message(FATAL_ERROR "cccl_add_executable requires SOURCES argument")
  endif()

  add_executable(${target_name} ${_cccl_SOURCES})
  cccl_configure_target(${target_name})

  if (_cccl_ADD_CTEST)
    add_test(NAME ${target_name} COMMAND "$<TARGET_FILE:${target_name}>")
  endif()

  if (NOT _cccl_NO_METATARGETS)
    set(metatarget_path ${target_name})
    if (DEFINED _cccl_METATARGET_PATH)
      set(metatarget_path ${_cccl_METATARGET_PATH})
    endif()
    cccl_ensure_metatargets(${target_name} METATARGET_PATH ${metatarget_path})
  endif()
endfunction()
