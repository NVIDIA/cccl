include(CheckSourceCompiles)
include(CMakePushCheckState)
cmake_push_check_state(RESET)
list(APPEND CMAKE_REQUIRED_INCLUDES "${libcudacxx_SOURCE_DIR}/include")
check_source_compiles(
  CUDA
  [[
    #include <cuda/std/detail/__config>

    #if defined(__CUDA_ARCH__) \
      && (!_CCCL_CTK_AT_LEAST(13, 5) || !_CCCL_HAS_NV_ATOMIC_BUILTINS())
    #  error "the NVVM atomic backend is unavailable"
    #endif

    __global__ void test() {}
    int main() { test<<<1, 1>>>(); }
  ]]
  libcudacxx_atomic_codegen_uses_nvvm_backend
)
cmake_pop_check_state()

if (libcudacxx_codegen_filecheck_atomics_sass)
  add_subdirectory(sass)
endif()

if (NOT libcudacxx_codegen_filecheck_atomics_ptx)
  return()
endif()

add_custom_target(libcudacxx.test.atomics.ptx)

if ("NVHPC" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
  return()
endif()

file(GLOB libcudacxx_atomic_codegen_tests CONFIGURE_DEPENDS "*.cu")
if (NOT libcudacxx_atomic_codegen_tests)
  return()
endif()

libcudacxx_codegen_check_tools(libcudacxx_codegen_tests_enabled)
if (NOT libcudacxx_codegen_tests_enabled)
  return()
endif()

libcudacxx_codegen_get_cuda_architectures(atomic_codegen_cuda_archs PTX)
if (NOT atomic_codegen_cuda_archs)
  message(
    STATUS
    "No compatible CUDA architectures requested; skipping atomic PTX codegen tests"
  )
  return()
endif()

foreach (arch IN LISTS atomic_codegen_cuda_archs)
  # For these kernel pointer arguments, NVVM specializes the operation to
  # global space on SM100+. On older targets both backends use generic PTX.
  if (libcudacxx_atomic_codegen_uses_nvvm_backend AND arch GREATER_EQUAL 100)
    set(address_prefix GLOBAL)
  else()
    set(address_prefix GENERIC)
  endif()
  if (arch GREATER_EQUAL 90)
    set(cluster_scope cluster)
  else()
    set(cluster_scope gpu)
  endif()

  libcudacxx_codegen_add_ptx_tests(
    AGGREGATE_TARGET libcudacxx.test.atomics.ptx
    TARGET_PREFIX atomic_codegen
    ARCH "${arch}"
    CHECK_PREFIXES SMXX
    OPTIONAL_CHECK_PREFIXES ${address_prefix}
    CHECK_DEFINITIONS CLUSTER_SCOPE=${cluster_scope}
    TESTS ${libcudacxx_atomic_codegen_tests}
    COMPILE_DEFINITIONS _CCCL_ATOMIC_UNSAFE_AUTOMATIC_STORAGE=1
  )
endforeach()
