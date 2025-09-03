include(${CMAKE_SOURCE_DIR}/benchmarks/cmake/CCCLBenchmarkRegistry.cmake)

cccl_get_nvbench()
cccl_get_nvbench_helper()

set(benches_root "${CMAKE_CURRENT_LIST_DIR}")

function(get_recursive_subdirs subdirs)
  set(dirs)
  file(GLOB_RECURSE contents
    CONFIGURE_DEPENDS
    LIST_DIRECTORIES ON
    "${CMAKE_CURRENT_LIST_DIR}/bench/*"
  )

  foreach(bench_dir IN LISTS contents)
    if(IS_DIRECTORY "${bench_dir}")
      list(APPEND dirs "${bench_dir}")
    endif()
  endforeach()

  set(${subdirs} "${dirs}" PARENT_SCOPE)
endfunction()

function(add_bench target_name bench_name bench_src)
  set(bench_target ${bench_name})
  set(${target_name} ${bench_target} PARENT_SCOPE)

  add_executable(${bench_target} "${bench_src}")
  cccl_configure_target(${bench_target} DIALECT 17)
  target_link_libraries(${bench_target} PRIVATE cccl.nvbench_helper nvbench::main)
endfunction()

function(add_bench_dir bench_dir)
  file(GLOB bench_srcs CONFIGURE_DEPENDS "${bench_dir}/*.cu")
  file(RELATIVE_PATH bench_prefix "${benches_root}" "${bench_dir}")
  file(TO_CMAKE_PATH "${bench_prefix}" bench_prefix)
  string(REPLACE "/" "." bench_prefix "${bench_prefix}")

  foreach(bench_src IN LISTS bench_srcs)
    foreach(cn_target IN LISTS cudax_TARGETS)
        cudax_get_target_property(config_prefix ${cn_target} PREFIX)

      get_filename_component(bench_name "${bench_src}" NAME_WLE)
      string(PREPEND bench_name "${config_prefix}.${bench_prefix}.")
      register_cccl_benchmark("${bench_name}" "")

      string(APPEND bench_name ".base")

      add_bench(base_bench_target ${bench_name} "${bench_src}")
      target_link_libraries(${bench_name} PRIVATE ${cn_target})
      cudax_clone_target_properties(${bench_name} ${cn_target})

      target_compile_options(${bench_name} PRIVATE "--extended-lambda")
    endforeach()
  endforeach()
endfunction()

get_recursive_subdirs(subdirs)

foreach(subdir IN LISTS subdirs)
  add_bench_dir("${subdir}")
endforeach()
