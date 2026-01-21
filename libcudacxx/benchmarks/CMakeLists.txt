include(${CMAKE_SOURCE_DIR}/benchmarks/cmake/CCCLBenchmarkRegistry.cmake)

cccl_get_nvbench_helper()

set(benches_root "${CMAKE_CURRENT_LIST_DIR}")

if (NOT CMAKE_BUILD_TYPE STREQUAL "Release")
  message(FATAL_ERROR "PSTL benchmarks must be built in release mode.")
endif()

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  message(
    FATAL_ERROR
    "CMAKE_CUDA_ARCHITECTURES must be set to build PSTL benchmarks."
  )
endif()

set(benches_meta_target libcudacxx.all.benches)
add_custom_target(${benches_meta_target})

function(get_recursive_subdirs subdirs)
  set(dirs)
  file(
    GLOB_RECURSE contents
    CONFIGURE_DEPENDS
    LIST_DIRECTORIES ON
    "${CMAKE_CURRENT_LIST_DIR}/bench/*"
  )

  foreach (test_dir IN LISTS contents)
    if (IS_DIRECTORY "${test_dir}")
      list(APPEND dirs "${test_dir}")
    endif()
  endforeach()

  set(${subdirs} "${dirs}" PARENT_SCOPE)
endfunction()

create_benchmark_registry()

function(add_bench target_name bench_name bench_src)
  set(bench_target ${bench_name})
  set(${target_name} ${bench_target} PARENT_SCOPE)

  add_executable(${bench_target} "${bench_src}")
  cccl_configure_target(${bench_target} DIALECT 17)
  target_link_libraries(
    ${bench_target}
    PRIVATE libcudacxx::libcudacxx cccl.nvbench_helper nvbench::main
  )
endfunction()

function(add_bench_dir bench_dir)
  file(GLOB bench_srcs CONFIGURE_DEPENDS "${bench_dir}/*.cu")
  file(RELATIVE_PATH bench_prefix "${benches_root}" "${bench_dir}")
  file(TO_CMAKE_PATH "${bench_prefix}" bench_prefix)
  string(REPLACE "/" "." bench_prefix "${bench_prefix}")

  foreach (bench_src IN LISTS bench_srcs)
    # base tuning
    get_filename_component(bench_name "${bench_src}" NAME_WLE)
    string(PREPEND bench_name "libcudacxx.${bench_prefix}.")

    set(base_bench_name "${bench_name}.base")
    add_bench(base_bench_target ${base_bench_name} "${bench_src}")
    add_dependencies(${benches_meta_target} ${base_bench_target})
    target_compile_definitions(${base_bench_target} PRIVATE TUNE_BASE=1)
    target_compile_options(${base_bench_target} PRIVATE "--extended-lambda")
    # benchmarking
    register_cccl_benchmark("${bench_name}" "")
  endforeach()
endfunction()

get_recursive_subdirs(subdirs)

foreach (subdir IN LISTS subdirs)
  add_bench_dir("${subdir}")
endforeach()
