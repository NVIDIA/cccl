cccl_get_cudatoolkit()

get_filename_component(nvrtc_cccl_source_dir "${Thrust_SOURCE_DIR}" DIRECTORY)

set(nvrtc_ctk_paths)
foreach (nvrtc_ctk_include_dir IN LISTS CUDAToolkit_INCLUDE_DIRS)
  string(APPEND nvrtc_ctk_paths "  \"-I${nvrtc_ctk_include_dir}\",\n")
endforeach()

configure_file(
  "${Thrust_SOURCE_DIR}/testing/cmake/nvrtc_args.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/nvrtc_args.h"
)

target_include_directories(${test_target} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")

target_link_libraries(${test_target} PRIVATE CUDA::nvrtc)
