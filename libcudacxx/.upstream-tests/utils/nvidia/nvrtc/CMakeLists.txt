project(nvrtcc)

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

add_executable(nvrtcc nvrtcc.cpp)

target_link_libraries(nvrtcc CUDA::nvrtc CUDA::cudart CUDA::cuda_driver)
target_compile_features(nvrtcc PRIVATE cxx_std_17)
