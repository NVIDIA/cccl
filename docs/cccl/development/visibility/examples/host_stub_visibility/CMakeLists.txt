project(DeveloperGuideHostStubVisibility CUDA CXX)

add_executable(host_stub_visibility main.cu)
add_library(host_stub_visibility_lib_a SHARED tu_a.cu)
add_library(host_stub_visibility_lib_b SHARED tu_b.cu)

target_link_libraries(
  host_stub_visibility
  PRIVATE host_stub_visibility_lib_a host_stub_visibility_lib_b
)
