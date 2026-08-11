cccl_get_cudatoolkit()

get_filename_component(nvrtc_cccl_source_dir "${Thrust_SOURCE_DIR}" DIRECTORY)
list(GET CUDAToolkit_INCLUDE_DIRS 0 nvrtc_ctk_include_dir)

configure_file(
  "${Thrust_SOURCE_DIR}/testing/cmake/nvrtc_args.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/nvrtc_args.h"
)

target_include_directories(${test_target} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")

target_link_libraries(${test_target} PRIVATE CUDA::nvrtc)
