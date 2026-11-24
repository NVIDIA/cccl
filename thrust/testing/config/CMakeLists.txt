# Compile tests for the deprecated host compiler warning in
# thrust/detail/config/config.h. See the sources for how and why the compiler
# detection is faked. See NVIDIA/cccl#1620.

foreach (thrust_target IN LISTS THRUST_TARGETS)
  thrust_get_target_property(config_device ${thrust_target} DEVICE)

  # The warning is backend independent. Only check it for the CPP device
  # backend, so that the tests do not require a CUDA compiler.
  if (NOT config_device STREQUAL "CPP")
    continue()
  endif()

  thrust_get_target_property(config_prefix ${thrust_target} PREFIX)

  # The warning has to reach the user. Note that cccl_add_xfail_compile_target_test
  # picks the target up from the caller's `test_target` variable, so it has to be
  # named that.
  set(test_target ${config_prefix}.test.config.deprecated_compiler_warning)
  cccl_add_executable(
    ${test_target}
    SOURCES deprecated_compiler_warning.cpp
    NO_METATARGETS
  )
  target_link_libraries(${test_target} PRIVATE ${thrust_target})
  cccl_add_xfail_compile_target_test(
    ${test_target}
    ERROR_REGEX "Thrust requires at least GCC"
  )

  # ... and CCCL_IGNORE_DEPRECATED_COMPILER has to silence it again. Warnings
  # are errors for CCCL targets, so compiling the target is the test.
  set(
    suppressed_target
    ${config_prefix}.test.config.deprecated_compiler_warning_suppressed
  )
  cccl_add_executable(
    ${suppressed_target}
    SOURCES deprecated_compiler_warning_suppressed.cpp
  )
  target_link_libraries(${suppressed_target} PRIVATE ${thrust_target})
endforeach()
