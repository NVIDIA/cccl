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
    self
    "${options}"
    "${oneValueArgs}"
    "${multiValueArgs}"
    ${ARGN}
  )
  cccl_parse_arguments_error_checks(
    "cccl_add_executable"
    ERROR_UNPARSED
    REQUIRED_VALUES SOURCES
    DEFAULT_VALUES METATARGET_PATH "${target_name}"
  )

  add_executable(${target_name} ${self_SOURCES})
  cccl_configure_target(${target_name})

  if (self_ADD_CTEST)
    add_test(NAME ${target_name} COMMAND "$<TARGET_FILE:${target_name}>")
  endif()

  if (NOT self_NO_METATARGETS)
    cccl_ensure_metatargets(
      ${target_name}
      METATARGET_PATH ${self_METATARGET_PATH}
    )
  endif()
endfunction()
