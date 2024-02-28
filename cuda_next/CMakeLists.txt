if (NOT DEFINED CCCL_ENABLE_CUDA_NEXT OR CCCL_ENABLE_CUDA_NEXT)
    cmake_minimum_required(VERSION 3.15)

    project(CUDA_NEXT VERSION 0.1)

    enable_language(CUDA)

    add_library(CUDA_Next INTERFACE)
    target_link_libraries(CUDA_Next INTERFACE libcudacxx::libcudacxx)
    target_include_directories(CUDA_Next INTERFACE "${CMAKE_CURRENT_LIST_DIR}/include")

    # TODO add the below to CCCL cmake 
    #target_link_libraries(CCCL::CCCL CUDA_Next)

    include(CTest)
    enable_testing()

    add_subdirectory(test)
endif()
