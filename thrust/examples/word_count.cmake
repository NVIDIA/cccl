target_compile_options(
  ${example_target}
  PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>: --extended-lambda>
)

# This check is actually not correct, because we must check the host compiler, not the CXX compiler.
# We rely on these usually being the same ;)
if (
  "Clang" STREQUAL "${CMAKE_CXX_COMPILER_ID}"
  AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 13
)
  # When clang >= 13 is used as host compiler, we get the following warning:
  #   nvcc_internal_extended_lambda_implementation:312:22: error: definition of implicit copy constructor for '__nv_hdl_wrapper_t<false, true, false, __nv_dl_tag<void (*)(), &TestAddressStabilityLambda, 2>, int (const int &)>' is deprecated because it has a user-declared copy assignment operator [-Werror,-Wdeprecated-copy]
  #   312 | __nv_hdl_wrapper_t & operator=(const __nv_hdl_wrapper_t &in) = delete;
  #       |                      ^
  # Let's suppress it until NVBug 4980157 is resolved.
  target_compile_options(
    ${example_target}
    PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>: -Wno-deprecated-copy>
  )
endif()
