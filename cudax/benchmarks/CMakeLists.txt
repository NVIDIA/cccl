include(${CMAKE_SOURCE_DIR}/benchmarks/cmake/CCCLBenchmarkRegistry.cmake)

cccl_get_nvbench()
cccl_get_nvbench_helper()

set(benches_root "${CMAKE_CURRENT_LIST_DIR}")

function(add_bench target_name bench_name bench_src)
  set(bench_target ${bench_name})
  set(${target_name} ${bench_target} PARENT_SCOPE)

  cccl_add_executable(${bench_target} SOURCES "${bench_src}")
  target_link_libraries(
    ${bench_target}
    PRIVATE #
      cccl.nvbench_helper
      nvbench::main
  )
endfunction()

# Creates one benchmark target per source file in `bench_srcs`. The created targets are
# returned in `bench_targets_var` so the caller can attach any extra dependencies its
# benchmarks need.
function(add_bench_dir bench_targets_var)
  set(bench_srcs ${ARGN})
  file(
    RELATIVE_PATH
    bench_prefix
    "${benches_root}"
    "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  file(TO_CMAKE_PATH "${bench_prefix}" bench_prefix)
  string(REPLACE "/" "." bench_prefix "${bench_prefix}")

  set(bench_targets)

  foreach (bench_src IN LISTS bench_srcs)
    get_filename_component(bench_name "${bench_src}" NAME_WLE)
    string(PREPEND bench_name "cudax.${bench_prefix}.")
    register_cccl_benchmark("${bench_name}" "")

    string(APPEND bench_name ".base")

    add_bench(base_bench_target ${bench_name} "${bench_src}")
    target_link_libraries(${bench_name} PRIVATE cudax.compiler_interface)
    target_compile_options(
      ${bench_name}
      PRIVATE
        "$<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--extended-lambda>"
        # cudax.compiler_interface enables assertions for tests/examples; benchmarks should measure release behavior.
        "$<$<COMPILE_LANGUAGE:CUDA>:-UCCCL_ENABLE_ASSERTIONS>"
        "$<$<COMPILE_LANGUAGE:CXX>:-UCCCL_ENABLE_ASSERTIONS>"
    )

    list(APPEND bench_targets "${bench_name}")
  endforeach()

  set(${bench_targets_var} "${bench_targets}" PARENT_SCOPE)
endfunction()

add_subdirectory(bench)
