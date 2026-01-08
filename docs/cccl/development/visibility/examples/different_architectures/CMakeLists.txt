project(DeveloperGuideDifferentArchitectures CUDA CXX)

add_library(different_architectures_lib_a SHARED tu_a.cu)
set_target_properties(
  different_architectures_lib_a
  PROPERTIES CUDA_ARCHITECTURES "86;90a"
)

add_library(different_architectures_lib_b SHARED tu_b.cu)
set_target_properties(
  different_architectures_lib_b
  PROPERTIES CUDA_ARCHITECTURES "75;86;90a"
)

add_executable(different_architectures main.cu)
set_target_properties(
  different_architectures
  PROPERTIES CUDA_ARCHITECTURES "75;86"
)

target_link_libraries(
  different_architectures
  PRIVATE different_architectures_lib_a different_architectures_lib_b
)

add_executable(different_architectures_switched main.cu kernel.cu)
set_target_properties(
  different_architectures_switched
  PROPERTIES CUDA_ARCHITECTURES "75;86"
)

target_link_libraries(
  different_architectures_switched
  PRIVATE different_architectures_lib_b different_architectures_lib_a
)
