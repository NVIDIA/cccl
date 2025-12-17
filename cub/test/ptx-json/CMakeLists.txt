if (NOT "${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
  message(STATUS "Skipping ptx-json tests on non-Linux platforms.")
  return()
endif()

if ("${CMAKE_CUDA_COMPILER_ID}" STREQUAL "Clang")
  message(STATUS "Skipping ptx-json tests for clang-cuda.")
  return()
endif()

if (CMAKE_CUDA_STANDARD LESS 20)
  message(
    STATUS
    "Skipping ptx-json tests; CMAKE_CUDA_STANDARD is less than 20."
  )
  return()
endif()

include(CheckIncludeFileCXX)
check_include_file_cxx("format" _CCCL_PTX_JSON_TEST_HAS_FORMAT)
mark_as_advanced(_CCCL_PTX_JSON_TEST_HAS_FORMAT)
if (NOT _CCCL_PTX_JSON_TEST_HAS_FORMAT)
  message(STATUS "Skipping ptx-json tests because <format> is not available.")
  return()
endif()

cccl_get_json()

set(filter_target "cub.test.ptx_json.filter")
cccl_add_executable(${filter_target} SOURCES filter.cpp)
target_link_libraries(
  ${filter_target}
  PRIVATE #
    cub.compiler_interface
    nlohmann_json::nlohmann_json
    CUDA::cudart
)

function(cub_detail_ptx_json_add_test target_name_var source)
  string(
    REGEX REPLACE
    "ptx_json_test_([^.]*).cu"
    "cub.test.ptx_json.\\1"
    target_name
    "${source}"
  )
  set(target_name_var ${target_name} PARENT_SCOPE)

  add_library(${target_name} OBJECT "${source}")
  cccl_configure_target(${target_name})
  cccl_ensure_metatargets(${target_name})
  target_link_libraries(${target_name} PRIVATE cub.compiler_interface)
  set_target_properties(
    ${target_name}
    PROPERTIES #
      CUDA_CUBIN_COMPILATION ON
      CUDA_ARCHITECTURES "80-real"
  )

  add_test(
    NAME ${target_name}
    # gersemi: off
    COMMAND
      "${CMAKE_CURRENT_SOURCE_DIR}/dump_and_check.bash"
        $<TARGET_FILE:${filter_target}>
        $<TARGET_OBJECTS:${target_name}>
        "${CMAKE_CURRENT_SOURCE_DIR}/${source}"
        "test-json-id"
    # gersemi: on
  )
endfunction()

file(
  GLOB test_srcs
  RELATIVE "${CMAKE_CURRENT_LIST_DIR}"
  CONFIGURE_DEPENDS
  ptx_json_test_*.cu
)

foreach (test_src IN LISTS test_srcs)
  cub_detail_ptx_json_add_test(test_target "${test_src}")
endforeach()

if (CUB_ENABLE_HEADER_TESTING)
  set(headertest_target cub.headers.ptx_json)
  cccl_generate_header_tests(
    ${headertest_target}
    cub
    GLOBS #
      "cub/detail/*ptx-json*"
      "cub/detail/ptx-json/*.cuh"
  )
  target_link_libraries(
    ${headertest_target}
    PUBLIC #
      cub.compiler_interface
      nlohmann_json::nlohmann_json
      CUDA::cudart
  )
endif()
