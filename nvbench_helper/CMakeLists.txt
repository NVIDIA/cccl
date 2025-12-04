cmake_minimum_required(VERSION 3.21)

project(nvbench_helper LANGUAGES CXX CUDA)

include(CMakeDependentOption)

cccl_get_catch2()
cccl_get_cub()
cccl_get_cudatoolkit()
cccl_get_libcudacxx()
cccl_get_nvbench()
cccl_get_thrust()

add_library(cccl.nvbench_helper OBJECT nvbench_helper/nvbench_helper.cu)
cccl_configure_target(cccl.nvbench_helper DIALECT 17)
target_link_libraries(
  cccl.nvbench_helper
  PUBLIC
    libcudacxx::libcudacxx
    CUB::CUB
    # This Thrust target does not define any host/device system. Those are added later if needed:
    Thrust::Thrust
    nvbench::nvbench
  PRIVATE #
    cccl.compiler_interface
    CUDA::curand
)

target_include_directories(
  cccl.nvbench_helper
  PUBLIC "${CMAKE_CURRENT_LIST_DIR}/nvbench_helper"
)

# Don't enable tests when pulled in as an internal dependency.
# Require CCCL_ENABLE_NVBENCH_HELPER to be explicitly enabled.
cmake_dependent_option(
  nvbench_helper_ENABLE_TESTING
  "Enable tests for nvbench_helper"
  ON
  CCCL_ENABLE_NVBENCH_HELPER
  OFF
)
mark_as_advanced(nvbench_helper_ENABLE_TESTING)

if (nvbench_helper_ENABLE_TESTING)
  cccl_get_boost()

  thrust_create_target(cccl.nvbench_helper.test.Thrust.cpp.cuda DEVICE CUDA)
  thrust_create_target(cccl.nvbench_helper.test.Thrust.cpp.cpp DEVICE CPP)

  function(add_nvbench_helper_test device_system)
    set(nvbench_helper_test_target nvbench_helper.test.${device_system})
    add_executable(
      ${nvbench_helper_test_target}
      test/gen_seed.cu
      test/gen_range.cu
      test/gen_entropy.cu
      test/gen_uniform_distribution.cu
      test/gen_power_law_distribution.cu
    )
    cccl_configure_target(${nvbench_helper_test_target} DIALECT 17)
    target_link_libraries(
      ${nvbench_helper_test_target}
      PRIVATE
        cccl.compiler_interface
        cccl.nvbench_helper
        cccl.nvbench_helper.test.Thrust.cpp.${device_system}
        Catch2::Catch2WithMain
        Boost::math
    )

    add_test(
      NAME ${nvbench_helper_test_target}
      COMMAND ${nvbench_helper_test_target}
    )
  endfunction()

  add_nvbench_helper_test(cpp)
  add_nvbench_helper_test(cuda)
endif()
