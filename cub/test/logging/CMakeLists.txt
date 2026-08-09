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

# Environment variable unset -> logging disabled, log() must stay silent.
add_test(
  NAME cub.test.logging.unset
  COMMAND
    "${CMAKE_COMMAND}" -E env --unset=CCCL_EXPERIMENTAL_LOGGING
    "$<TARGET_FILE:cub.test.logging>" --expect-disabled
)
set_tests_properties(
  cub.test.logging.unset
  PROPERTIES FAIL_REGULAR_EXPRESSION "${log_sentinel}"
)

# Environment variable set to 0 -> logging disabled, log() must stay silent
add_test(
  NAME cub.test.logging.zero
  COMMAND
    "${CMAKE_COMMAND}" -E env CCCL_EXPERIMENTAL_LOGGING=0
    "$<TARGET_FILE:cub.test.logging>" --expect-disabled
)
set_tests_properties(
  cub.test.logging.zero
  PROPERTIES FAIL_REGULAR_EXPRESSION "${log_sentinel}"
)

# Environment variable set to 1 -> logging enabled and log() emits its sentinel
add_test(
  NAME cub.test.logging.enabled
  COMMAND
    "${CMAKE_COMMAND}" -E env CCCL_EXPERIMENTAL_LOGGING=1
    "$<TARGET_FILE:cub.test.logging>" --expect-enabled
)
set_tests_properties(
  cub.test.logging.enabled
  PROPERTIES PASS_REGULAR_EXPRESSION "${log_sentinel}"
)

# log_always() must emit its sentinel regardless of the environment variable.
add_test(
  NAME cub.test.logging.always
  COMMAND
    "${CMAKE_COMMAND}" -E env --unset=CCCL_EXPERIMENTAL_LOGGING
    "$<TARGET_FILE:cub.test.logging>" --expect-disabled
)
set_tests_properties(
  cub.test.logging.always
  PROPERTIES PASS_REGULAR_EXPRESSION "${log_always_sentinel}"
)

# Compiled with CCCL_DISABLE_LOGGING: logging_enabled() is always false and neither log() nor
# log_always() produce any output, even with CCCL_EXPERIMENTAL_LOGGING=1.
add_test(
  NAME cub.test.logging.compile_disabled
  COMMAND
    "${CMAKE_COMMAND}" -E env CCCL_EXPERIMENTAL_LOGGING=1
    "$<TARGET_FILE:cub.test.logging.disabled>" --expect-disabled
)
set_tests_properties(
  cub.test.logging.compile_disabled
  PROPERTIES FAIL_REGULAR_EXPRESSION "${log_sentinel}|${log_always_sentinel}"
)

# Smoke test that a real algorithm dispatch actually logs when logging is enabled
cccl_add_executable(
  cub.test.logging.dispatch
  SOURCES "logging_dispatch_smoke.cu"
)
target_link_libraries(cub.test.logging.dispatch PRIVATE cub.compiler_interface)
# We must use rapids_test_add instead of cub_add_test because the latter compiles non-Catch2 tests with CUB_DEBUG_SYNC,
# which enables CUB_DEBUG_LOG, which in turn compiles out the new experimental logging
rapids_test_add(
        NAME cub.test.logging.dispatch
        COMMAND
        "${CMAKE_COMMAND}"
        "-DCCCL_SOURCE_DIR=${CCCL_SOURCE_DIR}"
        "-DTEST=$<TARGET_FILE:cub.test.logging.dispatch>"
        -P "${CUB_SOURCE_DIR}/test/run_test.cmake"
        GPUS 1
        PERCENT ${CCCL_TEST_GPU_PERCENT_SMALL}
)
set_tests_properties(
  cub.test.logging.dispatch
  PROPERTIES
    ENVIRONMENT "CCCL_EXPERIMENTAL_LOGGING=1;SCCACHE_NO_CACHE=1"
    PASS_REGULAR_EXPRESSION
      "Dispatching DeviceReduce on compute capability [0-9]+\\.[0-9]+ with tuning: ReducePolicy \\{"
)
