# Fetch nvbench
CPMAddPackage("gh:NVIDIA/nvbench#main")

add_library(nvbench_helper SHARED nvbench_helper.cuh nvbench_helper.cu)
target_link_libraries(nvbench_helper PUBLIC CUB::CUB 
                                            Thrust::Thrust 
                                            CUB::libcudacxx 
                                            nvbench::nvbench
                                     PRIVATE CUDA::curand)

target_include_directories(nvbench_helper PUBLIC "${CMAKE_CURRENT_LIST_DIR}")
