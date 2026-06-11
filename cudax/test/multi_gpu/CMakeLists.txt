set(TEST_PREFIX multi_gpu)

function(multi_gpu_get_nccl)
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/cmake")

  find_package(NCCL REQUIRED)
endfunction()

multi_gpu_get_nccl()

add_subdirectory(sort)
