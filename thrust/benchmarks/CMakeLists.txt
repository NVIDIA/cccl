include(${CMAKE_SOURCE_DIR}/benchmarks/cmake/CCCLBenchmarkRegistry.cmake)

if(NOT CCCL_ENABLE_CUB)
  message(FATAL_ERROR "Thrust benchmarks depend on CUB: set CCCL_ENABLE_CUB.")
endif()

if(NOT CUB_ENABLE_BENCHMARKS)
  message(FATAL_ERROR "Thrust benchmarks depend on CUB benchmarks: set CUB_ENABLE_BENCHMARKS.")
endif()

set(benches_root "${CMAKE_CURRENT_LIST_DIR}")

function(get_recursive_subdirs subdirs)
  set(dirs)
  file(GLOB_RECURSE contents
    CONFIGURE_DEPENDS
    LIST_DIRECTORIES ON
    "${CMAKE_CURRENT_LIST_DIR}/bench/*"
  )

  foreach(test_dir IN LISTS contents)
    if(IS_DIRECTORY "${test_dir}")
      list(APPEND dirs "${test_dir}")
    endif()
  endforeach()

  set(${subdirs} "${dirs}" PARENT_SCOPE)
endfunction()

function(add_bench target_name bench_name bench_src)
  set(bench_target ${bench_name})
  set(${target_name} ${bench_target} PARENT_SCOPE)

  add_executable(${bench_target} "${bench_src}")

  set_target_properties(${bench_target}
    PROPERTIES
      ARCHIVE_OUTPUT_DIRECTORY "${THRUST_LIBRARY_OUTPUT_DIR}"
      LIBRARY_OUTPUT_DIRECTORY "${THRUST_LIBRARY_OUTPUT_DIR}"
      RUNTIME_OUTPUT_DIRECTORY "${THRUST_EXECUTABLE_OUTPUT_DIR}"
      CUDA_STANDARD 17
      CXX_STANDARD 17)
  target_link_libraries(${bench_target} PRIVATE nvbench_helper nvbench::main)
endfunction()

function(thrust_wrap_bench_in_cpp cpp_file_var cu_file thrust_target)
  thrust_get_target_property(prefix ${thrust_target} PREFIX)
  set(wrapped_source_file "${cu_file}")
  set(cpp_file "${CMAKE_CURRENT_BINARY_DIR}/${prefix}/${cu_file}.cpp")
  configure_file("${Thrust_SOURCE_DIR}/cmake/wrap_source_file.cpp.in" "${cpp_file}")
  set(${cpp_file_var} "${cpp_file}" PARENT_SCOPE)
endfunction()

function(add_bench_dir bench_dir)
  file(GLOB bench_srcs CONFIGURE_DEPENDS "${bench_dir}/*.cu")
  file(RELATIVE_PATH bench_prefix "${benches_root}" "${bench_dir}")
  file(TO_CMAKE_PATH "${bench_prefix}" bench_prefix)
  string(REPLACE "/" "." bench_prefix "${bench_prefix}")

  foreach(bench_src IN LISTS bench_srcs)
    foreach(thrust_target IN LISTS THRUST_TARGETS)
      thrust_get_target_property(config_prefix ${thrust_target} PREFIX)
      thrust_get_target_property(config_device ${thrust_target} DEVICE)

      # Wrap the .cu file in .cpp for non-CUDA backends
      if ("CUDA" STREQUAL "${config_device}")
        set(real_bench_src "${bench_src}")
      else()
        thrust_wrap_bench_in_cpp(real_bench_src "${bench_src}" ${thrust_target})
      endif()

      get_filename_component(bench_name "${bench_src}" NAME_WLE)
      string(PREPEND bench_name "${config_prefix}.${bench_prefix}.")
      register_cccl_benchmark("${bench_name}" "")

      string(APPEND bench_name ".base")
      add_bench(base_bench_target ${bench_name} "${real_bench_src}")
      target_link_libraries(${bench_name} PRIVATE ${thrust_target})
      thrust_clone_target_properties(${bench_name} ${thrust_target})
    endforeach()
  endforeach()
endfunction()

get_recursive_subdirs(subdirs)

foreach(subdir IN LISTS subdirs)
  add_bench_dir("${subdir}")
endforeach()
