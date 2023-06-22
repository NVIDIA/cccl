find_package(CUDAToolkit REQUIRED)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

# Defer dependencies collection to nvbench helper 
add_subdirectory(nvbench_helper)

set(benches_root "${CMAKE_CURRENT_LIST_DIR}")

if(NOT CMAKE_BUILD_TYPE STREQUAL "Release")
  message(FATAL_ERROR "CUB benchmarks must be built in release mode.")
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  message(FATAL_ERROR "CMAKE_CUDA_ARCHITECTURES must be set to build CUB benchmarks.")
endif()

set(benches_meta_target cub.all.benches)
add_custom_target(${benches_meta_target})

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

set(meta_path "${CMAKE_BINARY_DIR}/cub_bench_meta.csv")
file(REMOVE "${meta_path}")

set(ctk_version "${CUDAToolkit_VERSION}")
message(STATUS "CTK version: ${ctk_version}")

find_package(Git REQUIRED)
if(GIT_FOUND)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} describe
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE cub_revision
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    message(STATUS "Git revision: ${cub_revision}")
else()
    message(WARNING "Git not found. Unable to determine Git revision.")
endif()

file(APPEND "${meta_path}" "ctk_version,${ctk_version}\n")
file(APPEND "${meta_path}" "cub_revision,${cub_revision}\n")

function(get_bench_ranges src bench_name)
  file(READ "${src}" file_data)
  set(param_regex "//[ ]+%RANGE%[ ]+([^ ]+)[ ]+([^ ]+)[ ]+([^\n]*)")

  string(REGEX MATCHALL "${param_regex}" matches "${file_data}")

  set(ranges "")

  foreach(match IN LISTS matches)
    string(REGEX MATCH "${param_regex}" unused "${match}")

    set(def ${CMAKE_MATCH_1})
    set(label ${CMAKE_MATCH_2})
    set(range ${CMAKE_MATCH_3})
    set(ranges "${ranges}${def}|${label}=${range},")

    string(REPLACE ":" ";" range "${range}")
    list(LENGTH range range_len)

    if (NOT "${range_len}" STREQUAL 3)
      message(FATAL_ERROR "Range should be represented as 'start:end:step'")
    endif()
  endforeach()

  string(LENGTH "${ranges}" ranges_length)
  math(EXPR last_character_index "${ranges_length} - 1")
  string(SUBSTRING "${ranges}" 0 ${last_character_index} ranges)
  file(APPEND "${meta_path}" "${bench_name},${ranges}\n")
endfunction()

function(add_bench target_name bench_name bench_src)
  set(bench_target ${bench_name})
  set(${target_name} ${bench_target} PARENT_SCOPE)

  add_executable(${bench_target} "${bench_src}")
  target_link_libraries(${bench_target} PRIVATE nvbench_helper nvbench::main)
  set_target_properties(${bench_target}
    PROPERTIES
      ARCHIVE_OUTPUT_DIRECTORY "${CUB_LIBRARY_OUTPUT_DIR}"
      LIBRARY_OUTPUT_DIRECTORY "${CUB_LIBRARY_OUTPUT_DIR}"
      RUNTIME_OUTPUT_DIRECTORY "${CUB_EXECUTABLE_OUTPUT_DIR}"
  )
endfunction()

function(add_bench_dir bench_dir)
  file(GLOB bench_srcs CONFIGURE_DEPENDS "${bench_dir}/*.cu")
  file(RELATIVE_PATH bench_prefix "${benches_root}" "${bench_dir}")
  file(TO_CMAKE_PATH "${bench_prefix}" bench_prefix)
  string(REPLACE "/" "." bench_prefix "${bench_prefix}")

  foreach(bench_src IN LISTS bench_srcs)
    # base tuning
    get_filename_component(bench_name "${bench_src}" NAME_WLE)
    string(PREPEND bench_name "cub.${bench_prefix}.")
    
    set(base_bench_name "${bench_name}.base")
    add_bench(base_bench_target ${base_bench_name} "${bench_src}")
    add_dependencies(${benches_meta_target} ${base_bench_target})
    target_compile_definitions(${base_bench_target} PRIVATE TUNE_BASE=1)

    # tuning
    if (CUB_ENABLE_TUNING)
      set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${bench_src}")
      get_bench_ranges("${bench_src}" "${bench_name}")
      set(tuning_name "${bench_name}.variant")
      set(tuning_path "${CMAKE_BINARY_DIR}/${tuning_name}.h")
      add_bench(bench_target ${tuning_name} "${bench_src}")
      file(WRITE "${tuning_path}" "#pragma once\n")
      target_compile_options(${bench_target} PRIVATE "-include${tuning_path}")
    endif()
  endforeach()
endfunction()

get_recursive_subdirs(subdirs)

foreach(subdir IN LISTS subdirs)
  add_bench_dir("${subdir}")
endforeach()
