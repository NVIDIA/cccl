cccl_get_c2h()

function(cccl_c_parallel_add_freestanding_test target_name_var source)
  get_filename_component(target_name "${source}" NAME_WE)
  string(
    REGEX REPLACE
    "test_([^.]*)"
    "cccl.c.parallel.test.\\1"
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
  target_link_libraries(${target_name} PRIVATE hostjit_lib CUDA::cudart)
endfunction()

file(
  GLOB freestanding_srcs
  RELATIVE "${CMAKE_CURRENT_LIST_DIR}"
  CONFIGURE_DEPENDS
  *.cpp
)

foreach (freestanding_src IN LISTS freestanding_srcs)
  cccl_c_parallel_add_freestanding_test(test_target "${freestanding_src}")
endforeach()
