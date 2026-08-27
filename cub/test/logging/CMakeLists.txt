# The basic logging tests do not launch any kernels and therefore do not require a GPU. They exercise
# cub::detail::logging_enabled(), cub::detail::log() and cub::detail::log_always() under different
# values of the CCCL_EXPERIMENTAL_LOGGING environment variable, as well as the CCCL_DISABLE_LOGGING
# compile-time switch. The test source lives in this subdirectory so it is excluded from the GPU
# test auto-glob in the parent CMakeLists.txt.

# Sentinels emitted by logging_test.cu. Keep in sync with that file.
set(log_sentinel "CCCL_LOG_SENTINEL")
set(log_always_sentinel "CCCL_LOG_ALWAYS_SENTINEL")

# Regular binary: logging is compiled in and gated at runtime by the environment variable.
cccl_add_executable(cub.test.logging SOURCES "logging_test.cu")
target_link_libraries(cub.test.logging PRIVATE cub.compiler_interface)

# Variant compiled with CCCL_DISABLE_LOGGING: all logging code is removed at compile time.
cccl_add_executable(cub.test.logging.disabled SOURCES "logging_test.cu")
target_link_libraries(cub.test.logging.disabled PRIVATE cub.compiler_interface)
target_compile_definitions(
  cub.test.logging.disabled
  PRIVATE CCCL_DISABLE_LOGGING
)

# Environment variable unset -> logging disabled: log() must stay silent, but log_always() must still fire.
add_test(
  NAME cub.test.logging.unset
  COMMAND
    "${CMAKE_COMMAND}" -E env --unset=CCCL_EXPERIMENTAL_LOGGING --
    "$<TARGET_FILE:cub.test.logging>" --expect-disabled
)
set_tests_properties(
  cub.test.logging.unset
  PROPERTIES
    FAIL_REGULAR_EXPRESSION "${log_sentinel}"
    PASS_REGULAR_EXPRESSION "${log_always_sentinel}"
)

# Environment variable set to 0 -> logging disabled: log() must stay silent, but log_always() must still fire.
add_test(
  NAME cub.test.logging.zero
  COMMAND
    "${CMAKE_COMMAND}" -E env CCCL_EXPERIMENTAL_LOGGING=0 --
    "$<TARGET_FILE:cub.test.logging>" --expect-disabled
)
set_tests_properties(
  cub.test.logging.zero
  PROPERTIES
    FAIL_REGULAR_EXPRESSION "${log_sentinel}"
    PASS_REGULAR_EXPRESSION "${log_always_sentinel}"
)

# Environment variable set to 1 -> logging enabled: log() emits its sentinel and log_always() fires too.
# The sentinels are printed on consecutive lines, so a single regex asserts both are present.
add_test(
  NAME cub.test.logging.enabled
  COMMAND
    "${CMAKE_COMMAND}" -E env CCCL_EXPERIMENTAL_LOGGING=1 --
    "$<TARGET_FILE:cub.test.logging>" --expect-enabled
)
set_tests_properties(
  cub.test.logging.enabled
  PROPERTIES PASS_REGULAR_EXPRESSION "${log_sentinel}.*${log_always_sentinel}"
)

# Compiled with CCCL_DISABLE_LOGGING: logging_enabled() is always false and neither log() nor
# log_always() produce any output, even with CCCL_EXPERIMENTAL_LOGGING=1.
add_test(
  NAME cub.test.logging.compile_disabled
  COMMAND
    "${CMAKE_COMMAND}" -E env CCCL_EXPERIMENTAL_LOGGING=1 --
    "$<TARGET_FILE:cub.test.logging.disabled>" --expect-disabled
)
set_tests_properties(
  cub.test.logging.compile_disabled
  PROPERTIES FAIL_REGULAR_EXPRESSION "${log_sentinel}|${log_always_sentinel}"
)

# Smoke test that a real algorithm dispatch actually logs when logging is enabled. We reuse
# cub_add_test() for consistency with the rest of the CUB test suite. That helper compiles non-Catch2
# tests with CUB_DEBUG_SYNC, which enables CUB_DEBUG_LOG and would compile out the new experimental
# logging, so we undo it again by removing the definition. We can't use a -UCUB_DEBUG_SYNC compile
# option instead: nvcc forwards it to the MSVC host compiler verbatim, and sccache fails to parse the
# resulting "-U" argument on Windows. If this stops working, the experimental logging line is not
# emitted and the PASS_REGULAR_EXPRESSION below fails the test.
cub_add_test(dispatch_test_target "logging.dispatch" "" "logging_dispatch_smoke.cu" 0)
cub_configure_cuda_target(${dispatch_test_target} RDC OFF)
get_target_property(
  _cub_logging_smoke_defs
  ${dispatch_test_target}
  COMPILE_DEFINITIONS
)
list(REMOVE_ITEM _cub_logging_smoke_defs CUB_DEBUG_SYNC)
set_property(
  TARGET ${dispatch_test_target}
  PROPERTY COMPILE_DEFINITIONS "${_cub_logging_smoke_defs}"
)
set_tests_properties(
  ${dispatch_test_target}
  PROPERTIES
    ENVIRONMENT "CCCL_EXPERIMENTAL_LOGGING=1"
    PASS_REGULAR_EXPRESSION
      "Dispatching DeviceReduce on compute capability [0-9]+\\.[0-9]+ with tuning: ReducePolicy \\{"
)
