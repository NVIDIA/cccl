cmake_minimum_required(VERSION 3.21)

project(nvbench_helper LANGUAGES CXX CUDA)

find_package(CCCL CONFIG REQUIRED
  NO_DEFAULT_PATH # Only check the explicit HINTS below:
  HINTS "${CCCL_SOURCE_DIR}/lib/cmake/cccl/"
)

cccl_get_catch2()
cccl_get_nvbench()

add_library(cccl.nvbench_helper OBJECT nvbench_helper/nvbench_helper.cu)

target_link_libraries(cccl.nvbench_helper
  PUBLIC
    libcudacxx::libcudacxx
    CUB::CUB
    # This Thrust target does not define any host/device system. Those are added later if needed:
    Thrust::Thrust
    nvbench::nvbench
  PRIVATE
    CUDA::curand
)

target_include_directories(cccl.nvbench_helper PUBLIC "${CMAKE_CURRENT_LIST_DIR}/nvbench_helper")
set_target_properties(cccl.nvbench_helper PROPERTIES CUDA_STANDARD 17 CXX_STANDARD 17)

option(nvbench_helper_ENABLE_TESTING "Enable tests for nvbench_helper" OFF)
mark_as_advanced(nvbench_helper_ENABLE_TESTING)

if (nvbench_helper_ENABLE_TESTING)
  cccl_get_boost()

  thrust_create_target(cccl.nvbench_helper.test.Thrust.cpp.cuda DEVICE CUDA)
  thrust_create_target(cccl.nvbench_helper.test.Thrust.cpp.cpp  DEVICE CPP)

  function(add_nvbench_helper_test device_system)
    set(nvbench_helper_test_target nvbench_helper.test.${device_system})
    add_executable(${nvbench_helper_test_target} test/gen_seed.cu
                                                 test/gen_range.cu
                                                 test/gen_entropy.cu
                                                 test/gen_uniform_distribution.cu
                                                 test/gen_power_law_distribution.cu)
    cccl_configure_target(${nvbench_helper_test_target} DIALECT 17)
    target_link_libraries(${nvbench_helper_test_target} PRIVATE
      cccl.nvbench_helper
      cccl.nvbench_helper.test.Thrust.cpp.${device_system}
      Catch2::Catch2WithMain
      Boost::math
    )
  endfunction()

  add_nvbench_helper_test(cpp)
  add_nvbench_helper_test(cuda)
endif()
