if(MSVC)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler\" /Zc:__cplusplus\"")
endif()

if ("Clang" STREQUAL "${CMAKE_CUDA_COMPILER_ID}")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-unknown-cuda-version -Xclang=-fcuda-allow-variadic-functions")
endif ()
