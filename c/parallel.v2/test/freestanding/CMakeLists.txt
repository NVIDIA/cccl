cccl_get_c2h()

function(cccl_c_parallel_v2_add_freestanding_test target_name_var source)
  get_filename_component(target_name "${source}" NAME_WE)
  cmake_parse_arguments(arg "LIBNVCC_ONLY" "" "" ${ARGN})
  string(
    REGEX REPLACE
    "test_([^.]*)"
    "cccl.c.parallel.v2.test.freestanding.\\1"
    target_name
    "${target_name}"
  )
  set(${target_name_var} ${target_name} PARENT_SCOPE)

  cccl_add_executable(
    ${target_name}
    ADD_CTEST
    NO_METATARGETS
    DIALECT 20
    SOURCES "${source}"
  )
  # LIBNVCC_ONLY keeps hostjit_lib off the link line, for a test that drives the
  # library through its own API the way an unrelated consumer would.
  if (arg_LIBNVCC_ONLY)
    set(test_libs libnvcc ${CMAKE_DL_LIBS})
  else()
    set(test_libs cccl.c.parallel.v2.hostjit_lib)
  endif()

  target_link_libraries(
    ${target_name}
    PRIVATE ${test_libs} CUDA::cudart
  )
endfunction()

file(
  GLOB freestanding_srcs
  RELATIVE "${CMAKE_CURRENT_LIST_DIR}"
  CONFIGURE_DEPENDS
  *.cpp
)

set(freestanding_libnvcc_only_srcs test_libnvcc_artifact_unload.cpp)

# These measure free device memory before and after an unload, which is a property of
# the whole device: another test allocating beside them moves the reading, and the
# module they are watching for is 16 MiB against a slack of one. So they run alone.
set(freestanding_serial_srcs
    test_hostjit_unload.cpp
    test_libnvcc_artifact_unload.cpp
)

foreach (freestanding_src IN LISTS freestanding_srcs)
  set(src_options)
  if (freestanding_src IN_LIST freestanding_libnvcc_only_srcs)
    set(src_options LIBNVCC_ONLY)
  endif()

  cccl_c_parallel_v2_add_freestanding_test(test_target "${freestanding_src}" ${src_options})

  if (freestanding_src IN_LIST freestanding_serial_srcs)
    set_tests_properties(${test_target} PROPERTIES RUN_SERIAL TRUE)
  endif()
endforeach()
