##===----------------------------------------------------------------------===##
##
## Part of libcu++ in the CUDA C++ Core Libraries,
## under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
## SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
##
##===----------------------------------------------------------------------===##

option(
  LIBCUDACXX_REQUIRE_CODEGEN_TEST_TOOLS
  "Fail configuration when tools required by libcu++ codegen tests are missing."
  OFF
)

function(libcudacxx_codegen_check_tools out_var)
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
    if (LIBCUDACXX_REQUIRE_CODEGEN_TEST_TOOLS)
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

# Return the concrete CUDA architectures requested for a PTX or SASS test.
# CMake records the architectures detected for "native" separately, while
# "all" and "all-major" have compiler-specific expansions.
function(libcudacxx_codegen_get_cuda_architectures out_var code_kind)
  if (NOT code_kind MATCHES "^(PTX|SASS)$")
    message(FATAL_ERROR "Unsupported codegen test kind: ${code_kind}")
  endif()

  set(requested_architectures)
  foreach (arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
    if (arch STREQUAL "native")
      # Native selects an architecture, rather than a particular output kind.
      # Strip CMake's -real suffix so both PTX and SASS tests use that arch.
      foreach (native_arch IN LISTS CMAKE_CUDA_ARCHITECTURES_NATIVE)
        if (native_arch MATCHES "^([0-9]+[af]?)")
          list(APPEND requested_architectures "${CMAKE_MATCH_1}")
        endif()
      endforeach()
    elseif (arch STREQUAL "all")
      list(APPEND requested_architectures ${CMAKE_CUDA_ARCHITECTURES_ALL})
    elseif (arch STREQUAL "all-major")
      list(APPEND requested_architectures ${CMAKE_CUDA_ARCHITECTURES_ALL_MAJOR})
    else()
      list(APPEND requested_architectures "${arch}")
    endif()
  endforeach()

  set(cuda_architectures)
  foreach (arch IN LISTS requested_architectures)
    if (arch MATCHES "^([0-9]+[af]?)(-(real|virtual))?$")
      set(output_kind "${CMAKE_MATCH_3}")
      if (code_kind STREQUAL "PTX" AND output_kind STREQUAL "real")
        continue()
      endif()
      if (code_kind STREQUAL "SASS" AND output_kind STREQUAL "virtual")
        continue()
      endif()
      list(APPEND cuda_architectures "${CMAKE_MATCH_1}")
    endif()
  endforeach()
  list(REMOVE_DUPLICATES cuda_architectures)
  set(${out_var} "${cuda_architectures}" PARENT_SCOPE)
endfunction()

function(libcudacxx_codegen_set_cuda_arch target_name arch)
  if (arch MATCHES "[af]$")
    set_target_properties(${target_name} PROPERTIES CUDA_ARCHITECTURES OFF)
    target_compile_options(
      ${target_name}
      PRIVATE
        "--generate-code=arch=compute_${arch},code=[compute_${arch},sm_${arch}]"
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
  string(REGEX REPLACE "[^A-Za-z0-9_]" "_" check_suffix "${check_prefixes}")
  set(check_target_name "${target_path}.${check_suffix}.check")

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
        ${ARGN}
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
  )
  set(multi_value_args CHECK_PREFIXES COMPILE_DEFINITIONS)
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

  set(dump_options)
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
    list(APPEND dump_options --dump-sass)
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
      ${dump_options}
    )
  endforeach()
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
  set(multi_value_args ARCHITECTURES TESTS COMPILE_DEFINITIONS)
  cmake_parse_arguments(
    arg
    "${options}"
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN}
  )

  foreach (test_path IN LISTS arg_TESTS)
    file(READ "${test_path}" test_contents)
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

      libcudacxx_codegen_add_test(
        ${test_options}
        AGGREGATE_TARGET ${arg_AGGREGATE_TARGET}
        TARGET_PREFIX ${arg_TARGET_PREFIX}
        CODE_KIND sass
        ARCH ${arch}
        TEST "${test_path}"
        CHECK_PREFIXES "${check_prefixes}"
        COMPILE_DEFINITIONS ${arg_COMPILE_DEFINITIONS}
      )
    endforeach()
  endforeach()
endfunction()
