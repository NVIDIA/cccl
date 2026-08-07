##===----------------------------------------------------------------------===##
##
## Part of libcu++ in the CUDA C++ Core Libraries,
## under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
## SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
##
##===----------------------------------------------------------------------===##

include("${CMAKE_CURRENT_LIST_DIR}/../../../cmake/CCCLTestParams.cmake")

option(
  LIBCUDACXX_REQUIRE_CODEGEN_TEST_TOOLS
  "Fail configuration when tools required by libcu++ codegen tests are missing."
  OFF
)

function(libcudacxx_codegen_check_tools out_var)
  if (LIBCUDACXX_REQUIRE_CODEGEN_TEST_TOOLS)
    set(require_codegen_tools TRUE)
  else()
    set(require_codegen_tools FALSE)
  endif()

  get_property(
    tools_available
    GLOBAL
    PROPERTY libcudacxx_codegen_tools_available
  )
  if (NOT "${tools_available}" STREQUAL "")
    set(${out_var} "${tools_available}" PARENT_SCOPE)
    return()
  endif()

  find_program(
    libcudacxx_codegen_filecheck
    NAMES
      FileCheck
      FileCheck-22
      FileCheck-21
      FileCheck-20
      FileCheck-19
      FileCheck-18
  )
  find_program(libcudacxx_codegen_cuobjdump NAMES cuobjdump)
  find_program(libcudacxx_codegen_bash NAMES bash)

  set(missing_tools)
  if (NOT libcudacxx_codegen_filecheck)
    list(APPEND missing_tools FileCheck)
  endif()
  if (NOT libcudacxx_codegen_cuobjdump)
    list(APPEND missing_tools cuobjdump)
  endif()
  if (NOT libcudacxx_codegen_bash)
    list(APPEND missing_tools bash)
  endif()

  if (missing_tools)
    list(JOIN missing_tools ", " missing_tools)
    if (require_codegen_tools)
      message(
        FATAL_ERROR
        "Tools required by libcu++ codegen tests were not found: ${missing_tools}"
      )
    endif()

    message(
      STATUS
      "Codegen test tools not found (${missing_tools}); skipping libcu++ codegen tests"
    )
    set(tools_available FALSE)
  else()
    message(STATUS "Codegen test tools found; enabling libcu++ codegen tests")
    set(tools_available TRUE)
  endif()

  set_property(
    GLOBAL
    PROPERTY libcudacxx_codegen_tools_available "${tools_available}"
  )
  set(${out_var} "${tools_available}" PARENT_SCOPE)
endfunction()

set(
  libcudacxx_codegen_dump_and_check
  "${CMAKE_CURRENT_LIST_DIR}/../codegen/dump_and_check.bash"
)

# Return the concrete CUDA architectures requested for codegen testing.
# Virtual-only architectures do not produce the SASS consumed by these tests;
# special values such as "native" and "all" do not identify a stable FileCheck
# prefix and should be replaced with explicit architectures by the caller.
function(libcudacxx_codegen_get_cuda_architectures out_var)
  set(cuda_architectures)
  foreach (arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
    if (arch MATCHES "^([0-9]+[af]?)(-real)?$")
      list(APPEND cuda_architectures "${CMAKE_MATCH_1}")
    endif()
  endforeach()
  list(REMOVE_DUPLICATES cuda_architectures)

  if (NOT cuda_architectures)
    message(
      FATAL_ERROR
      "Libcu++ FileCheck codegen tests require at least one explicit real CUDA architecture; got '${CMAKE_CUDA_ARCHITECTURES}'"
    )
  endif()

  set(${out_var} "${cuda_architectures}" PARENT_SCOPE)
endfunction()

function(libcudacxx_codegen_set_cuda_arch target_name arch)
  if (arch MATCHES "[af]$")
    set_target_properties(${target_name} PROPERTIES CUDA_ARCHITECTURES OFF)
    target_compile_options(
      ${target_name}
      PRIVATE "--generate-code=arch=compute_${arch},code=sm_${arch}"
    )
  else()
    set_target_properties(
      ${target_name}
      PROPERTIES CUDA_ARCHITECTURES "${arch}"
    )
  endif()
endfunction()

# Given the test file content, extract the SM archs that are checked for.
# SMXX:       any arch
# SM1XX:      any SM100-series arch
# SM100-PLUS: SM100 and newer
function(
  libcudacxx_codegen_get_sass_check_prefixes
  out_prefixes
  out_has_specific_checks
  test_contents
  arch
)
  set(check_prefixes SMXX)
  set(arch_prefix "SM${arch}")

  string(
    REGEX MATCH
    "; ${arch_prefix}(:|-LABEL:|-NOT:|-NEXT:|-SAME:|-DAG:|-COUNT:|-EMPTY:)"
    has_arch_prefix
    "${test_contents}"
  )
  if (has_arch_prefix)
    list(APPEND check_prefixes "${arch_prefix}")
  endif()

  if (arch MATCHES "^1[0-9][0-9][af]?$")
    string(
      REGEX MATCH
      "; SM1XX(:|-LABEL:|-NOT:|-NEXT:|-SAME:|-DAG:|-COUNT:|-EMPTY:)"
      has_sm1xx_prefix
      "${test_contents}"
    )
    if (has_sm1xx_prefix)
      list(APPEND check_prefixes SM1XX)
    endif()
  endif()

  string(
    REGEX MATCHALL
    "; SM[0-9]+-PLUS(:|-[A-Z]+:)"
    plus_prefixes
    "${test_contents}"
  )
  if ("${arch}" MATCHES "[af]$")
    set(plus_prefixes)
  endif()
  foreach (plus_prefix IN LISTS plus_prefixes)
    string(REGEX REPLACE ".*SM([0-9]+)-PLUS.*" "\\1" plus_arch "${plus_prefix}")
    if (arch GREATER_EQUAL plus_arch)
      list(APPEND check_prefixes "SM${plus_arch}-PLUS")
    endif()
  endforeach()
  list(REMOVE_DUPLICATES check_prefixes)
  list(JOIN check_prefixes "," check_prefixes)

  string(
    REGEX MATCH
    "; SM([0-9]+[a-f]?|1XX|[0-9]+-PLUS)(:|-[A-Z]+:)"
    has_specific_checks
    "${test_contents}"
  )
  set(${out_prefixes} "${check_prefixes}" PARENT_SCOPE)
  set(${out_has_specific_checks} "${has_specific_checks}" PARENT_SCOPE)
endfunction()

function(
  libcudacxx_codegen_add_check_target
  target_path
  target_name
  test_path
  check_prefixes
)
  cmake_parse_arguments(arg "" "DUMP_MODE" "CHECK_DEFINITIONS" ${ARGN})

  string(REGEX REPLACE "[^A-Za-z0-9_]" "_" check_suffix "${check_prefixes}")
  set(check_target_name "${target_path}.${check_suffix}.check")

  set(filecheck_definitions)
  foreach (definition IN LISTS arg_CHECK_DEFINITIONS)
    list(APPEND filecheck_definitions "-D${definition}")
  endforeach()

  add_custom_target(
    ${check_target_name}
    DEPENDS ${target_name}
    # gersemi: off
    COMMAND
      "${CMAKE_COMMAND}" -E env
        "CUOBJDUMP=${libcudacxx_codegen_cuobjdump}"
        "FILECHECK=${libcudacxx_codegen_filecheck}"
        "${libcudacxx_codegen_bash}"
        "${libcudacxx_codegen_dump_and_check}"
        $<TARGET_FILE:${target_name}>
        "${test_path}"
        "${check_prefixes}"
        "${arg_DUMP_MODE}"
        ${filecheck_definitions}
    # gersemi: on
  )
  cccl_ensure_metatargets(${check_target_name})
endfunction()

function(libcudacxx_codegen_add_test)
  set(options SEPARABLE_COMPILATION)
  set(
    one_value_args
    AGGREGATE_TARGET
    TARGET_PREFIX
    CODE_KIND
    ARCH
    TEST
    VARIANT
  )
  set(
    multi_value_args
    CHECK_PREFIXES
    CHECK_DEFINITIONS
    COMPILE_DEFINITIONS
  )
  cmake_parse_arguments(
    arg
    "${options}"
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN}
  )

  cmake_path(GET arg_TEST FILENAME test_file)
  cmake_path(REMOVE_EXTENSION test_file LAST_ONLY OUTPUT_VARIABLE test_name)
  set(test_target_path "${arg_AGGREGATE_TARGET}.sm${arg_ARCH}.${test_name}")
  set(
    target_name
    "${arg_TARGET_PREFIX}_${arg_CODE_KIND}_sm${arg_ARCH}_${test_name}"
  )
  if (arg_VARIANT)
    string(APPEND test_target_path ".${arg_VARIANT}")
    string(APPEND target_name ".${arg_VARIANT}")
  endif()

  add_library(${target_name} STATIC "${arg_TEST}")
  libcudacxx_codegen_set_cuda_arch(${target_name} "${arg_ARCH}")
  target_compile_options(${target_name} PRIVATE "-Wno-comment")
  target_include_directories(
    ${target_name}
    PRIVATE "${libcudacxx_SOURCE_DIR}/include"
  )

  if (arg_COMPILE_DEFINITIONS)
    target_compile_definitions(
      ${target_name}
      PRIVATE ${arg_COMPILE_DEFINITIONS}
    )
  endif()

  set(dump_mode --dump-ptx)
  if (arg_CODE_KIND STREQUAL "ptx")
    # Clang stopped emitting PTX in clang20. Add flags to re-enable it.
    if (
      CMAKE_CUDA_COMPILER_ID STREQUAL Clang
      AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 20
    )
      target_compile_options(
        ${target_name}
        PRIVATE "--cuda-include-ptx=sm_${arg_ARCH}"
      )
    endif()
  else()
    set(dump_mode --dump-sass)
    if (arg_SEPARABLE_COMPILATION)
      # CMake supplies the compile phase, so request relocatable device code
      # without adding a second compile-phase option via -dc.
      set_property(TARGET ${target_name} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    endif()
  endif()

  foreach (check_prefix IN LISTS arg_CHECK_PREFIXES)
    libcudacxx_codegen_add_check_target(
      ${test_target_path}
      ${target_name}
      "${arg_TEST}"
      "${check_prefix}"
      DUMP_MODE "${dump_mode}"
      CHECK_DEFINITIONS ${arg_CHECK_DEFINITIONS}
    )
  endforeach()
endfunction()

# Separate variant definitions used as codegen test metadata from ordinary
# definitions. FILECHECK_PREFIX_<name>=<prefix> adds the upper-case form of
# <prefix> to the FileCheck invocation for that variant; all other definitions
# are available to both the compiler and FileCheck.
function(
  libcudacxx_codegen_get_variant_options
  out_compile_definitions
  out_check_definitions
  out_check_prefixes
)
  set(compile_definitions)
  set(check_definitions)
  set(check_prefixes)

  foreach (definition IN LISTS ARGN)
    if (definition MATCHES "^FILECHECK_PREFIX_[A-Za-z0-9_]+=(.*)$")
      set(check_prefix "${CMAKE_MATCH_1}")
      if (NOT check_prefix MATCHES "^[A-Za-z][A-Za-z0-9_-]*$")
        message(
          FATAL_ERROR
          "Invalid FileCheck prefix '${check_prefix}' in '${definition}'"
        )
      endif()
      string(TOUPPER "${check_prefix}" check_prefix)
      list(APPEND check_prefixes "${check_prefix}")
    else()
      list(APPEND compile_definitions "${definition}")
      list(APPEND check_definitions "${definition}")
    endif()
  endforeach()

  list(REMOVE_DUPLICATES check_prefixes)
  set(${out_compile_definitions} "${compile_definitions}" PARENT_SCOPE)
  set(${out_check_definitions} "${check_definitions}" PARENT_SCOPE)
  set(${out_check_prefixes} "${check_prefixes}" PARENT_SCOPE)
endfunction()

function(libcudacxx_codegen_add_ptx_tests)
  set(options)
  set(one_value_args AGGREGATE_TARGET TARGET_PREFIX ARCH)
  set(multi_value_args CHECK_PREFIXES TESTS COMPILE_DEFINITIONS)
  cmake_parse_arguments(
    arg
    "${options}"
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN}
  )

  foreach (test_path IN LISTS arg_TESTS)
    libcudacxx_codegen_add_test(
      AGGREGATE_TARGET ${arg_AGGREGATE_TARGET}
      TARGET_PREFIX ${arg_TARGET_PREFIX}
      CODE_KIND ptx
      ARCH ${arg_ARCH}
      TEST "${test_path}"
      CHECK_PREFIXES ${arg_CHECK_PREFIXES}
      COMPILE_DEFINITIONS ${arg_COMPILE_DEFINITIONS}
    )
  endforeach()
endfunction()

function(libcudacxx_codegen_add_sass_tests)
  set(options)
  set(one_value_args AGGREGATE_TARGET TARGET_PREFIX)
  set(
    multi_value_args
    ARCHITECTURES
    CHECK_PREFIXES
    TESTS
    COMPILE_DEFINITIONS
  )
  cmake_parse_arguments(
    arg
    "${options}"
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN}
  )

  foreach (test_path IN LISTS arg_TESTS)
    file(READ "${test_path}" test_contents)
    cccl_parse_variant_params(
      "${test_path}"
      num_variants
      variant_labels
      variant_definitions
    )
    cccl_log_variant_params(
      "${test_path}"
      ${num_variants}
      variant_labels
      variant_definitions
    )
    string(
      REGEX MATCH
      "; SM[0-9]+[af]-PLUS(:|-[A-Z]+:)"
      invalid_plus_prefix
      "${test_contents}"
    )
    if (invalid_plus_prefix)
      message(
        FATAL_ERROR
        "${test_path}: architecture- and family-specific SASS prefixes cannot use -PLUS: ${invalid_plus_prefix}"
      )
    endif()

    set(test_archs)
    foreach (arch IN LISTS arg_ARCHITECTURES)
      libcudacxx_codegen_get_sass_check_prefixes(
        check_prefixes
        has_arch_specific_checks
        "${test_contents}"
        "${arch}"
      )
      if (NOT "${check_prefixes}" STREQUAL "SMXX")
        list(APPEND test_archs "${arch}")
      endif()
    endforeach()

    if (NOT test_archs)
      if (has_arch_specific_checks)
        message(
          STATUS
          "-- Skipping ${test_path}: requires an unsupported CUDA architecture"
        )
        continue()
      endif()
      set(test_archs ${arg_ARCHITECTURES})
    endif()

    string(FIND "${test_contents}" "__device__" has_device_function)
    set(test_options)
    if (NOT has_device_function EQUAL -1)
      list(APPEND test_options SEPARABLE_COMPILATION)
    endif()

    foreach (arch IN LISTS test_archs)
      libcudacxx_codegen_get_sass_check_prefixes(
        check_prefixes
        has_arch_specific_checks
        "${test_contents}"
        "${arch}"
      )
      string(REPLACE "," ";" common_check_prefixes "${check_prefixes}")
      foreach (check_prefix IN LISTS arg_CHECK_PREFIXES)
        string(
          REGEX MATCH
          "; ${check_prefix}(:|-[A-Z]+:)"
          has_check_prefix
          "${test_contents}"
        )
        if (has_check_prefix)
          list(APPEND common_check_prefixes "${check_prefix}")
        endif()
      endforeach()
      list(REMOVE_DUPLICATES common_check_prefixes)

      if (num_variants EQUAL 0)
        list(JOIN common_check_prefixes "," combined_check_prefixes)
        libcudacxx_codegen_add_test(
          ${test_options}
          AGGREGATE_TARGET ${arg_AGGREGATE_TARGET}
          TARGET_PREFIX ${arg_TARGET_PREFIX}
          CODE_KIND sass
          ARCH ${arch}
          TEST "${test_path}"
          CHECK_PREFIXES "${combined_check_prefixes}"
          COMPILE_DEFINITIONS ${arg_COMPILE_DEFINITIONS}
        )
      else()
        math(EXPR last_variant "${num_variants} - 1")
        foreach (variant_index RANGE ${last_variant})
          cccl_get_variant_data(
            variant_labels
            variant_definitions
            ${variant_index}
            variant_label
            definitions
          )
          libcudacxx_codegen_get_variant_options(
            variant_compile_definitions
            variant_check_definitions
            variant_check_prefixes
            ${definitions}
          )

          set(combined_check_prefixes ${common_check_prefixes})
          list(APPEND combined_check_prefixes ${variant_check_prefixes})
          list(REMOVE_DUPLICATES combined_check_prefixes)
          list(JOIN combined_check_prefixes "," combined_check_prefixes)

          libcudacxx_codegen_add_test(
            ${test_options}
            AGGREGATE_TARGET ${arg_AGGREGATE_TARGET}
            TARGET_PREFIX ${arg_TARGET_PREFIX}
            CODE_KIND sass
            ARCH ${arch}
            TEST "${test_path}"
            VARIANT "${variant_label}"
            CHECK_PREFIXES "${combined_check_prefixes}"
            CHECK_DEFINITIONS ${variant_check_definitions}
            COMPILE_DEFINITIONS
              ${arg_COMPILE_DEFINITIONS}
              ${variant_compile_definitions}
          )
        endforeach()
      endif()
    endforeach()
  endforeach()
endfunction()
