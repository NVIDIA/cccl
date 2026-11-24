if (NOT "${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
  message(STATUS "Skipping cubin-check tests on non-Linux platforms.")
  return()
endif()

if ("${CMAKE_CUDA_COMPILER_ID}" STREQUAL "Clang")
  message(STATUS "Skipping cubin-check tests for clang-cuda.")
  return()
endif()

find_program(filecheck "FileCheck")

if (NOT filecheck)
  message(STATUS "Skipping cubin-check tests because FileCheck was not found.")
  return()
endif()

find_program(cuobjdump "cuobjdump" REQUIRED)

function(cub_add_cubin_check_test source)
  string(
    REGEX REPLACE
    "cubin_check_test_([^.]*).cu"
    "cub.test.cubin_check.\\1"
    target_name
    "${source}"
  )

  add_library(${target_name} OBJECT "${source}")
  cccl_configure_target(${target_name})
  cccl_ensure_metatargets(${target_name})
  target_link_libraries(${target_name} PRIVATE cub.compiler_interface)
  set_target_properties(
    ${target_name}
    PROPERTIES #
      CUDA_CUBIN_COMPILATION ON
      # Some of our presets specify multiple architectures, so the hope here is to make the compilation a little faster.
      # Given that empty kernel doesn't depend on the arch, that should be fine.
      CUDA_ARCHITECTURES "80-real"
  )

  add_test(
    NAME ${target_name}
    # gersemi: off
    COMMAND
      "${CMAKE_CURRENT_SOURCE_DIR}/dump_and_check.bash"
        "${cuobjdump}"
        $<TARGET_OBJECTS:${target_name}>
        "${CMAKE_CURRENT_SOURCE_DIR}/${source}"
    # gersemi: on
  )
endfunction()

file(
  GLOB test_srcs
  RELATIVE "${CMAKE_CURRENT_LIST_DIR}"
  CONFIGURE_DEPENDS
  cubin_check_test_*.cu
)

foreach (test_src IN LISTS test_srcs)
  cub_add_cubin_check_test("${test_src}")
endforeach()
