# Fetch nvbench
CPMAddPackage("gh:NVIDIA/nvbench#main")

add_library(nvbench_helper OBJECT nvbench_helper.cuh nvbench_helper.cu)
target_link_libraries(nvbench_helper PUBLIC CUB::CUB 
                                            Thrust::Thrust 
                                            CUB::libcudacxx 
                                            nvbench::nvbench
                                     PRIVATE CUDA::curand)

target_include_directories(nvbench_helper PUBLIC "${CMAKE_CURRENT_LIST_DIR}")
set_target_properties(nvbench_helper PROPERTIES CUDA_STANDARD 17 CXX_STANDARD 17)
