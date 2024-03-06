if (TARGET CUDA_Next)
  return()
endif()

# We don't ship with CTK, so no need for the ugly name (yet)
add_library(CUDA_Next INTERFACE)
target_compile_features(CUDA_Next INTERFACE cxx_std_17)

target_include_directories(CUDA_Next INTERFACE ${CMAKE_CURRENT_LIST_DIR}/../../../include)
