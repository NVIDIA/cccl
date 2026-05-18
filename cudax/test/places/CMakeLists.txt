set(
  places_test_sources
  include_only.cu
  data_place_alloc.cu
  data_place_vmm.cu
  exec_place_scope.cu
  stream_pool.cu
)

set(places_fail_tests exec_place_scope_data_place_fail.cu)

set(
  places_unittested_headers
  cuda/experimental/__places/places.cuh
  cuda/experimental/__places/exec/green_context.cuh
  cuda/experimental/__places/partitions/blocked_partition.cuh
  cuda/experimental/__places/partitions/cyclic_shape.cuh
  cuda/experimental/__places/partitions/tiled_partition.cuh
)

cccl_get_cudatoolkit()

## cudax_add_places_test
#
# Add a places test executable and register it with ctest.
#
# target_name_var: Variable name to overwrite with the name of the test
#   target. Useful for adding target information after creation.
# source: The source file for the test.
#
function(cudax_add_places_test target_name_var source)
  get_filename_component(dir ${source} DIRECTORY)
  get_filename_component(filename ${source} NAME_WE)
  if (dir)
    set(filename "${dir}/${filename}")
  endif()
  string(REPLACE "/" "." test_name "${filename}")

  set(test_target cudax.test.places.${test_name})

  cccl_add_executable(${test_target} SOURCES ${source} ADD_CTEST)
  cudax_places_configure_target(${test_target})
  target_link_libraries(${test_target} PRIVATE cudax.compiler_interface)

  set(${target_name_var} ${test_target} PARENT_SCOPE)
endfunction()

## cudax_add_places_unittest_header
#
# Add a places unittested header executable and register it with ctest.
#
# Unittested headers contain a set of tests that are enabled by including
# `unittest.cuh` and defining `UNITTESTED_FILE`.
#
# target_name_var: Variable name to overwrite with the name of the test
#   target. Useful for adding target information after creation.
# source: The source file for the test.
#
function(cudax_add_places_unittest_header target_name_var source)
  get_filename_component(relative_path ${source} DIRECTORY)
  get_filename_component(filename ${source} NAME_WE)

  string(
    REPLACE
    "cuda/experimental/"
    ""
    test_label
    "${relative_path}/${filename}"
  )
  string(REPLACE "/" "." test_label "${test_label}")

  set(test_target "cudax.test.places.unittest_headers.${test_label}")

  get_filename_component(
    source_full_path
    ../../../cudax/include/${source}
    ABSOLUTE
  )
  set(source ${source_full_path})

  set(ut_template "${cudax_SOURCE_DIR}/cmake/places_header_unittest.in.cu")
  set(ut_source "${cudax_BINARY_DIR}/unittest_headers/${test_target}.cu")
  configure_file(${ut_template} ${ut_source} @ONLY)

  cccl_add_executable(${test_target} SOURCES ${ut_source} ADD_CTEST)
  cudax_places_configure_target(${test_target})
  target_link_libraries(${test_target} PRIVATE cudax.compiler_interface)

  set(${target_name_var} ${test_target} PARENT_SCOPE)
endfunction()

# Basic tests:
foreach (source IN LISTS places_test_sources)
  cudax_add_places_test(test_target "${source}")
endforeach()

# Unittested headers
foreach (source IN LISTS places_unittested_headers)
  cudax_add_places_unittest_header(test_target "${source}")
endforeach()

## cudax_add_places_fail_test
#
# Adds an EXCLUDE_FROM_ALL build target for `source` and a ctest that
# verifies compilation fails with errors matching expected-error annotations
# in the source file.
#
function(cudax_add_places_fail_test target_name_var source)
  get_filename_component(filename ${source} NAME_WE)

  set(test_target cudax.test.places.error.${filename})

  cccl_add_executable(
    ${test_target}
    SOURCES ${source}
    NO_METATARGETS
    NO_CLANG_TIDY
  )
  cudax_places_configure_target(${test_target})
  target_link_libraries(${test_target} PRIVATE cudax.compiler_interface)
  cccl_add_xfail_compile_target_test(
    ${test_target}
    SOURCE_FILE "${source}"
    ERROR_REGEX_LABEL "expected-error"
  )

  set(${target_name_var} ${test_target} PARENT_SCOPE)
endfunction()

# Expected-failure compile tests (_fail.cu convention)
foreach (source IN LISTS places_fail_tests)
  cudax_add_places_fail_test(test_target "${source}")
endforeach()
