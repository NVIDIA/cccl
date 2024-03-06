if ("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_LIST_DIR}")
  set(CUDA_NEXT_TOPLEVEL_PROJECT ON)
endif()

if (CCCL_ENABLE_CUDA_NEXT OR CUDA_NEXT_TOPLEVEL_PROJECT)
    cmake_minimum_required(VERSION 3.15)

    project(CUDA_NEXT VERSION 0.1)

    enable_language(CUDA)

    add_library(CUDA_Next INTERFACE)
    target_compile_features(CUDA_Next INTERFACE cxx_std_17)

    target_link_libraries(CUDA_Next INTERFACE libcudacxx::libcudacxx)
    target_include_directories(CUDA_Next INTERFACE "${CMAKE_CURRENT_LIST_DIR}/include")

    option(CUDA_NEXT_ENABLE_TESTING "Build CUDA Next tests." ON)
    if (CUDA_NEXT_ENABLE_TESTING)
        enable_testing()
        add_subdirectory(test)
    endif()

endif()
