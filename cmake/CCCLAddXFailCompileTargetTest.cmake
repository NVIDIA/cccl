# cccl_add_xfail_compile_target_test(
#   <target_name>
#   [TEST_NAME <test_name>]
#   [ERROR_REGEX <regex>]
#   [SOURCE_FILE <source_file>]
#   [ERROR_REGEX_LABEL <error_string>]
#   [ERROR_NUMBER <error_number>]
#   [ERROR_NUMBER_TARGET_NAME_REGEX <regex>]
# )
#
# Given a configured build target that is expected to fail to compile:
# - Mark the target as excluded from the `all` target.
# - Create a CTest test that compiles the target. If TEST_NAME is provided, it is used.
#   Otherwise, the target_name is used as the test name.
# - When the test runs, it passes if exactly one of the following conditions is met:
#   - A provided / detected error regex matches the compilation output, ignoring exit code.
#   - No error regex is provided / detected, and the compilation fails.
#
# An error regex may be explicitly provided via ERROR_REGEX, or it may be
# detected by scanning the SOURCE_FILE for a specially formatted comment.
#
# If ERROR_REGEX_LABEL is provided, the SOURCE_FILE will read, looking for a comment of the form:
#
# // <ERROR_REGEX_LABEL> {{"error_regex"}}
#
# An error number may be appended to the ERROR_REGEX_LABEL in the comment:
#
# // <ERROR_REGEX_LABEL>-<error_number> {{"error_regex"}}
#
# If ERROR_NUMBER_TARGET_NAME_REGEX is specified, the regex is used to capture
# the error_number from the target name. If target_name is
# "cccl.test.my_test.err_5.foo_3" and ERROR_NUMBER_TARGET_NAME_REGEX is
# "\\.err_([0-9]+)", the captured error number "5."
#
# // <ERROR_REGEX_LABEL>-<captured_error_number> {{"error_regex"}}
#
# If ERROR_NUMBER is provided, ERROR_NUMBER_TARGET_NAME_REGEX is ignored.
# If ERROR_NUMBER_TARGET_NAME_REGEX is provided but does not match, a plain ERROR_REGEX_LABEL is used.
#
# If both SOURCE_FILE and ERROR_REGEX_LABEL are provided, the source file will be added to the
# current directory's CMAKE_CONFIGURE_DEPENDS to ensure that changes to the file will re-trigger CMake.
function(cccl_add_xfail_compile_target_test target_name)
  set(options)
  set(
    oneValueArgs
    TEST_NAME
    ERROR_REGEX
    SOURCE_FILE
    ERROR_REGEX_LABEL
    ERROR_NUMBER
    ERROR_NUMBER_TARGET_NAME_REGEX
  )
  set(multiValueArgs)
  cmake_parse_arguments(
    self
    "${options}"
    "${oneValueArgs}"
    "${multiValueArgs}"
    ${ARGN}
  )
  cccl_parse_arguments_error_checks(
    "cccl_add_xfail_compile_target_test"
    ERROR_UNPARSED
    DEFAULT_VALUES #
      TEST_NAME ${target_name}
  )

  set(regex)
  if (DEFINED self_ERROR_REGEX)
    set(regex "${self_ERROR_REGEX}")
  elseif (DEFINED self_SOURCE_FILE AND DEFINED self_ERROR_REGEX_LABEL)
    set(error_label_regex "${self_ERROR_REGEX_LABEL}")

    # Cache all error label matches (with and without error numbers) as global properties.
    # This avoids re-reading and re-parsing the source file multiple times if multiple
    # tests are added for the same source file. Properties are used instead of cache variables
    # to ensure that the source is not cached in between CMake executions.
    get_filename_component(src_absolute "${self_SOURCE_FILE}" ABSOLUTE)
    string(MD5 source_filename_md5 "${src_absolute}")
    set(error_cache_property "_cccl_xfail_error_cache_${source_filename_md5}")
    get_property(error_cache_set GLOBAL PROPERTY "${error_cache_property}" SET)
    if (error_cache_set)
      get_property(error_cache GLOBAL PROPERTY "${error_cache_property}")
    else()
      file(READ "${src_absolute}" source_contents)
      string(
        REGEX MATCHALL
        "//[ \t]*${error_label_regex}(-[0-9]+)?[ \t]*{{\"([^\"]+)\"}}"
        error_cache
        "${source_contents}"
      )
      set_property(GLOBAL PROPERTY "${error_cache_property}" "${error_cache}")
    endif()

    # Changes to the source file should re-run CMake to pick-up new error specs:
    set_property(
      DIRECTORY
      APPEND
      PROPERTY CMAKE_CONFIGURE_DEPENDS "${src_absolute}"
    )

    set(error_number)
    if (DEFINED self_ERROR_NUMBER)
      set(error_number "${self_ERROR_NUMBER}")
    elseif (DEFINED self_ERROR_NUMBER_TARGET_NAME_REGEX)
      string(
        REGEX MATCH
        "${self_ERROR_NUMBER_TARGET_NAME_REGEX}"
        matched
        ${target_name}
      )
      if (matched)
        set(error_number "${CMAKE_MATCH_1}")
      endif()
    endif()

    # Look for a labeled error with the specific error number.
    if (NOT "${error_number}" STREQUAL "") # Check strings to allow "0"
      string(
        REGEX MATCH
        "//[ \t]*${error_label_regex}-${error_number}[ \t]*{{\"([^\"]+)\"}}"
        matched
        "${error_cache}"
      )
      if (matched)
        set(regex "${CMAKE_MATCH_1}")
      endif()
    endif()

    if (NOT regex)
      # Look for a labeled error without an error number.
      string(
        REGEX MATCH
        "//[ \t]*${error_label_regex}[ \t]*{{\"([^\"]+)\"}}"
        matched
        "${error_cache}"
      )
      if (matched)
        set(regex "${CMAKE_MATCH_1}")
      endif()
    endif()
  endif()

  message(VERBOSE "CCCL: Adding XFAIL test: ${self_TEST_NAME}")
  if (regex)
    message(VERBOSE "CCCL:   with expected regex: '${regex}'")
  endif()

  set_target_properties(${target_name} PROPERTIES EXCLUDE_FROM_ALL true)

  # The same target may be reused for multiple tests, and the output file
  # may exist if using a regex to check for warnings. Add a setup fixture to
  # delete the output file before each test run.
  if (NOT TEST ${target_name}.clean)
    add_test(
      NAME ${target_name}.clean
      # gersemi: off
      COMMAND
        "${CMAKE_COMMAND}" -E rm -f
          "$<TARGET_FILE:${target_name}>"
          "$<TARGET_OBJECTS:${target_name}>"
      # gersemi: on
    )
    set_tests_properties(
      ${target_name}.clean
      PROPERTIES FIXTURES_SETUP ${target_name}.clean
    )
  endif()

  add_test(
    NAME ${self_TEST_NAME}
    # gersemi: off
    COMMAND
      "${CMAKE_COMMAND}"
        --build "${CMAKE_BINARY_DIR}"
        --target ${target_name}
        --config $<CONFIGURATION>
    # gersemi: on
  )
  set_tests_properties(
    ${self_TEST_NAME}
    PROPERTIES FIXTURES_CLEANUP ${target_name}.clean
  )

  if (regex)
    set_tests_properties(
      ${self_TEST_NAME}
      PROPERTIES PASS_REGULAR_EXPRESSION "${regex}"
    )
  else()
    set_tests_properties(${self_TEST_NAME} PROPERTIES WILL_FAIL true)
  endif()
endfunction()
