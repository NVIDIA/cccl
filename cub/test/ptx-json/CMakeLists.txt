if (NOT "${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
  message(STATUS "Skipping ptx-json tests on non-Linux platforms.")
  return()
endif()

set(is_20_available FALSE)
foreach(cub_target IN LISTS CUB_TARGETS)
  cub_get_target_property(dialect ${cub_target} DIALECT)
  if (${dialect} GREATER_EQUAL 20)
    set(is_20_available TRUE)
    break()
  endif()
endforeach()

if(NOT is_20_available)
  message(STATUS "Skipping ptx-json tests because no targets with DIALECT >= 20 were found.")
  return()
endif()

# Remove -G, -g, and -lineinfo from CUDA_FLAGS, otherwise the location information
# will throw off the JSON output. This will only apply to targets created in this directory.
string(REGEX REPLACE "(-G|-g|-lineinfo)" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")

include(CheckIncludeFileCXX)
check_include_file_cxx("format" _CCCL_PTX_JSON_TEST_HAS_FORMAT)
mark_as_advanced(_CCCL_PTX_JSON_TEST_HAS_FORMAT)
if(NOT _CCCL_PTX_JSON_TEST_HAS_FORMAT)
  message(STATUS "Skipping ptx-json tests because <format> is not available.")
  return()
endif()

cccl_get_json()

add_executable(ptx-json-test-filter
  filter.cpp
)
cccl_configure_target(ptx-json-test-filter DIALECT 20)
target_link_libraries(ptx-json-test-filter
  nlohmann_json::nlohmann_json
  CUDA::cudart
  Thrust::Thrust
  CUB::CUB
)

function(cub_detail_ptx_json_add_test target_name_var source)
  foreach(cub_target IN LISTS CUB_TARGETS)
    cub_get_target_property(dialect ${cub_target} DIALECT)
    if (${dialect} LESS 20)
      continue()
    endif()

    cub_get_target_property(prefix ${cub_target} PREFIX)

    string(REGEX REPLACE "ptx_json_test_([^.]*).cu" "${prefix}.detail.ptx_json.test.\\1.ptx" target_name "${source}")
    set(target_name_var ${target_name} PARENT_SCOPE)

    add_library(${target_name} OBJECT
      "${source}"
    )

    cub_clone_target_properties(${target_name} ${cub_target})
    target_link_libraries(${target_name}
      PRIVATE ${cub_target}
    )
    set_target_properties(${target_name} PROPERTIES
      CUDA_PTX_COMPILATION ON
      # Use compute_80 -- CTK 13.0 started running ptxas on 90+, even when just producing PTX.
      # This breaks the ptx-json stuff, which produces intentionally invalid PTX.
      CUDA_ARCHITECTURES "80-virtual"
    )

    add_test(NAME ${target_name}
      COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/dump_and_check.bash
        $<TARGET_FILE:ptx-json-test-filter> $<TARGET_OBJECTS:${target_name}>
        ${CMAKE_CURRENT_SOURCE_DIR}/${source} "test-json-id"
    )
  endforeach()
endfunction()

file(GLOB test_srcs
  RELATIVE "${CMAKE_CURRENT_LIST_DIR}"
  CONFIGURE_DEPENDS
  ptx_json_test_*.cu
)

foreach(test_src IN LISTS test_srcs)
  cub_detail_ptx_json_add_test(test_target "${test_src}")
endforeach()
