find_package(CUDAToolkit REQUIRED)

set(cccl_revision "")
find_package(Git)
if(GIT_FOUND)
  execute_process(
      COMMAND ${GIT_EXECUTABLE} describe
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      OUTPUT_VARIABLE cccl_revision
      OUTPUT_STRIP_TRAILING_WHITESPACE)

  if(cccl_revision STREQUAL "")
    # Sometimes, there is no tag (shallow copy)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE cccl_revision
        OUTPUT_STRIP_TRAILING_WHITESPACE)
  endif()
endif()

# Sometimes this script is used outside of a Git repository.
# In this case, we read the revision from cccl/cccl_version instead.
if ("${cccl_revision}" STREQUAL "")
  file(READ "${CMAKE_SOURCE_DIR}/cccl_version" cccl_revision)
  string(STRIP "${cccl_revision}" cccl_revision)
  string(REPLACE "\n" "" cccl_revision "${cccl_revision}")
endif()
message(STATUS "Git revision: ${cccl_revision}")

function(get_meta_path meta_path)
  set(meta_path "${CMAKE_BINARY_DIR}/cccl_meta_bench.csv" PARENT_SCOPE)
endfunction()

function(create_benchmark_registry)
  get_meta_path(meta_path)

  set(ctk_version "${CUDAToolkit_VERSION}")
  message(STATUS "CTK version: ${ctk_version}")

  file(REMOVE "${meta_path}")
  file(APPEND "${meta_path}" "ctk_version,${ctk_version}\n")
  file(APPEND "${meta_path}" "cccl_revision,${cccl_revision}\n")
endfunction()

function(register_cccl_tuning bench_name ranges)
  get_meta_path(meta_path)
  if ("${ranges}" STREQUAL "")
    file(APPEND "${meta_path}" "${bench_name}\n")
  else()
    file(APPEND "${meta_path}" "${bench_name},${ranges}\n")
  endif()
endfunction()

function(register_cccl_benchmark bench_name)
  register_cccl_tuning("${bench_name}" "")
endfunction()
