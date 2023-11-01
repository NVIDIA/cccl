project(nvrtcc)

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

add_executable(libcudacxx.nvrtcc nvrtcc.cpp)

set_target_properties(libcudacxx.nvrtcc PROPERTIES OUTPUT_NAME nvrtcc)
target_link_libraries(libcudacxx.nvrtcc CUDA::nvrtc CUDA::cudart CUDA::cuda_driver)
target_compile_features(libcudacxx.nvrtcc PRIVATE cxx_std_17)
