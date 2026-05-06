project(DeveloperGuideDeviceKernelVisibility CUDA CXX)

add_executable(device_kernel_visibility main.cu)
add_library(device_kernel_visibility_liba SHARED tu_a.cu)
add_library(device_kernel_visibility_lib_b SHARED tu_b.cu)

target_link_libraries(
  device_kernel_visibility
  PRIVATE device_kernel_visibility_lib_a device_kernel_visibility_lib_b
)
