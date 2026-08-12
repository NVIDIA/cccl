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
  set(target_name_var ${target_name} PARENT_SCOPE)

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
    target_link_libraries(
      ${target_name}
      PRIVATE libnvcc CUDA::cudart ${CMAKE_DL_LIBS}
    )
  else()
    target_link_libraries(
      ${target_name}
      PRIVATE cccl.c.parallel.v2.hostjit_lib CUDA::cudart
    )
  endif()
endfunction()

file(
  GLOB freestanding_srcs
  RELATIVE "${CMAKE_CURRENT_LIST_DIR}"
  CONFIGURE_DEPENDS
  *.cpp
)

set(freestanding_libnvcc_only_srcs test_libnvcc_artifact_unload.cpp)

foreach (freestanding_src IN LISTS freestanding_srcs)
  if (freestanding_src IN_LIST freestanding_libnvcc_only_srcs)
    cccl_c_parallel_v2_add_freestanding_test(
      test_target
      "${freestanding_src}"
      LIBNVCC_ONLY
    )
  else()
    cccl_c_parallel_v2_add_freestanding_test(test_target "${freestanding_src}")
  endif()
endforeach()
