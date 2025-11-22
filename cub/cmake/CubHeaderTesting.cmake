# For every public header, build a translation unit containing `#include <header>`
# to let the compiler try to figure out warnings in that header if it is not otherwise
# included in tests, and also to verify if the headers are modular enough.
# .inl files are not globbed for, because they are not supposed to be used as public
# entrypoints.

function(cub_add_header_test label definitions)
  set(headertest_target cub.headers.${label})

  cccl_generate_header_tests(
    ${headertest_target}
    cub
    GLOBS "cub/*.cuh"
    # These headers have additional dependencies and strict compiler reqs.
    # They're effectively an implementation detail of cccl.c.parallel and
    # have their own testing.
    EXCLUDES #
      "cub/detail/*ptx-json*"
      "cub/detail/ptx-json/*.cuh"
  )
  cub_configure_cuda_target(${headertest_target} RDC ${CUB_FORCE_RDC})
  target_link_libraries(${headertest_target} PUBLIC cub.compiler_interface)
  target_compile_definitions(${headertest_target} PRIVATE ${definitions})
endfunction()

# Wrap Thrust/CUB in a custom namespace to check proper use of ns macros:
set(
  header_definitions
  "THRUST_WRAPPED_NAMESPACE=wrapped_thrust"
  "CUB_WRAPPED_NAMESPACE=wrapped_cub"
)
cub_add_header_test(base "${header_definitions}")

# Check that BF16 support can be disabled
set(
  header_definitions
  "THRUST_WRAPPED_NAMESPACE=wrapped_thrust"
  "CUB_WRAPPED_NAMESPACE=wrapped_cub"
  "CCCL_DISABLE_BF16_SUPPORT"
)
cub_add_header_test(no_bf16 "${header_definitions}")

# Check that half support can be disabled
set(
  header_definitions
  "THRUST_WRAPPED_NAMESPACE=wrapped_thrust"
  "CUB_WRAPPED_NAMESPACE=wrapped_cub"
  "CCCL_DISABLE_FP16_SUPPORT"
)
cub_add_header_test(no_half "${header_definitions}")

# Check that half support can be disabled
set(
  header_definitions
  "THRUST_WRAPPED_NAMESPACE=wrapped_thrust"
  "CUB_WRAPPED_NAMESPACE=wrapped_cub"
  "CCCL_DISABLE_FP8_SUPPORT"
)
cub_add_header_test(no_fp8 "${header_definitions}")
