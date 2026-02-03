option(
  LIBCUDACXX_TEST_WITH_NVRTC
  "Test libcu++ with runtime compilation instead of offline compilation. Only runs device side tests."
  OFF
)

###############################################################################
### C2H tests:
cccl_get_c2h()

set(c2h_all_target "libcudacxx.test.c2h_all")
add_custom_target(${c2h_all_target})

if (NOT LIBCUDACXX_TEST_WITH_NVRTC)
  file(
    GLOB_RECURSE test_srcs
    RELATIVE "${CMAKE_CURRENT_LIST_DIR}"
    CONFIGURE_DEPENDS
    *.cu
  )

  function(libcudacxx_add_test target_name_var source)
    string(REPLACE "/" "." target_name "${source}")
    string(PREPEND target_name "libcudacxx.test.")
    string(REGEX REPLACE "\\.[^.]+$" "" target_name "${target_name}")
    set(${target_name_var} ${target_name} PARENT_SCOPE)

    add_executable(${target_name} "${source}")
    cccl_configure_target(${target_name} DIALECT ${CMAKE_CUDA_STANDARD})
    target_include_directories(
      ${target_name}
      PRIVATE
        "${libcudacxx_SOURCE_DIR}/test/libcudacxx/cuda/ccclrt/common"
        "${libcudacxx_SOURCE_DIR}/test/support"
    )
    target_link_libraries(
      ${target_name}
      PRIVATE #
        libcudacxx.compiler_interface
        cccl.c2h.main
    )

    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      target_compile_options(${target_name} PRIVATE "-Wno-attributes")
    endif()

    add_dependencies(${c2h_all_target} ${target_name})

    add_test(NAME ${target_name} COMMAND ${target_name})
  endfunction()

  foreach (test_src IN LISTS test_srcs)
    libcudacxx_add_test(test_target "${test_src}")
  endforeach()
endif()

###############################################################################
### Lit tests:
macro(pythonize_bool var)
  if (${var})
    set(${var} True)
  else()
    set(${var} False)
  endif()
endmacro()

cccl_get_cudatoolkit()
cccl_get_dlpack()

get_target_property(CUDA_INCLUDE_DIR CUDA::cudart INTERFACE_INCLUDE_DIRECTORIES)

message(STATUS "Lit enabled CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

if (LIBCUDACXX_TEST_WITH_NVRTC)
  # TODO: Use project properties to get path to binary.
  # Should also set up dependency on the project when NVRTC is enabled
  foreach (include IN ITEMS ${CUDA_INCLUDE_DIR})
    string(APPEND LIBCUDACXX_TEST_COMPILER_FLAGS " -I'${include}'")
  endforeach()
  set(
    LIBCUDACXX_CUDA_COMPILER
    "${CMAKE_BINARY_DIR}/libcudacxx/test/utils/nvidia/nvrtc/nvrtcc"
  )
  set(LIBCUDACXX_CUDA_COMPILER_ARG1 "")
  set(LIBCUDACXX_CUDA_TEST_WITH_NVRTC "True")
  # Use the NVRTCC utility to run the built test outputs
  set(
    LIBCUDACXX_EXECUTOR
    "PrefixExecutor(['${LIBCUDACXX_CUDA_COMPILER}'], LocalExecutor())"
  )
  # Enable 128-bit types for NVRTC
  string(APPEND LIBCUDACXX_TEST_COMPILER_FLAGS " -device-int128")
  string(APPEND LIBCUDACXX_TEST_COMPILER_FLAGS " -device-float128")
else() # NOT LIBCUDACXX_TEST_WITH_NVRTC
  set(
    LIBCUDACXX_FORCE_INCLUDE
    "-include ${libcudacxx_SOURCE_DIR}/test/libcudacxx/force_include.h"
  )
  set(LIBCUDACXX_CUDA_COMPILER "${CMAKE_CUDA_COMPILER}")
  set(LIBCUDACXX_CUDA_TEST_WITH_NVRTC "False")
endif()

# enable exceptions and assertions in tests
string(APPEND LIBCUDACXX_TEST_COMPILER_FLAGS " -DCCCL_ENABLE_ASSERTIONS")
string(APPEND LIBCUDACXX_TEST_COMPILER_FLAGS " -I ${dlpack_SOURCE_DIR}/include")

# enable optional<T&>
string(APPEND LIBCUDACXX_TEST_COMPILER_FLAGS " -DCCCL_ENABLE_OPTIONAL_REF")

# Disable dialect deprecation
string(
  APPEND LIBCUDACXX_TEST_COMPILER_FLAGS
  " -DCCCL_IGNORE_DEPRECATED_CPP_DIALECT"
)
string(
  APPEND LIBCUDACXX_TEST_COMPILER_FLAGS
  " -DLIBCUDACXX_IGNORE_DEPRECATED_ABI"
)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  string(APPEND LIBCUDACXX_TEST_LINKER_FLAGS " -latomic")
endif()

if (NOT MSVC AND NOT ${CMAKE_CUDA_COMPILER_ID} STREQUAL "Clang")
  set(
    LIBCUDACXX_WARNING_LEVEL
    "--compiler-options=-Wall --compiler-options=-Wextra"
  )
endif()

if (MSVC)
  # We want to use cudaLaunchKernelEx which is guarded by __cplusplus
  if ("${CMAKE_CUDA_COMPILER_VERSION}" LESS "12.3.0")
    string(APPEND LIBCUDACXX_TEST_COMPILER_FLAGS " -Xcompiler=/Zc:__cplusplus")
  endif()

  # Require the conforming preprocessor
  string(APPEND LIBCUDACXX_TEST_COMPILER_FLAGS " -Xcompiler=/Zc:preprocessor")
  if (MSVC_TOOLSET_VERSION LESS 143)
    # winbase.h(9572): warning C5105: macro expansion producing 'defined' has undefined behavior
    string(APPEND LIBCUDACXX_TEST_COMPILER_FLAGS " -Xcompiler=/wd5105")
  endif()
endif()

if (${CMAKE_CUDA_COMPILER_ID} STREQUAL "Clang")
  string(
    APPEND LIBCUDACXX_TEST_COMPILER_FLAGS
    " ${CMAKE_CUDA_FLAGS}"
    " -Xclang -fcuda-allow-variadic-functions"
    " -Xclang -Wno-unused-parameter"
    " -Wno-unknown-cuda-version"
    " ${LIBCUDACXX_FORCE_INCLUDE}"
    " -I${libcudacxx_SOURCE_DIR}/include"
    " ${LIBCUDACXX_WARNING_LEVEL}"
  )

  string(
    APPEND LIBCUDACXX_TEST_LINKER_FLAGS
    " ${CMAKE_CUDA_FLAGS}"
    " -L${CUDAToolkit_LIBRARY_DIR}"
    " -lcuda"
    " -lcudart"
  )
elseif (${CMAKE_CUDA_COMPILER_ID} STREQUAL "NVIDIA")
  string(
    APPEND LIBCUDACXX_TEST_COMPILER_FLAGS
    " ${LIBCUDACXX_FORCE_INCLUDE}"
    " ${LIBCUDACXX_WARNING_LEVEL}"
    " -Wno-deprecated-gpu-targets"
  )
elseif (${CMAKE_CUDA_COMPILER_ID} STREQUAL "NVHPC")
  string(APPEND LIBCUDACXX_TEST_COMPILER_FLAGS " -stdpar")
  string(APPEND LIBCUDACXX_TEST_LINKER_FLAGS " -stdpar")
endif()

include(AddLLVM)

set(LIBCUDACXX_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")

set(
  LIBCUDACXX_TARGET_INFO
  "libcudacxx.test.target_info.LocalTI"
  CACHE STRING
  "TargetInfo to use when setting up test environment."
)
set(
  LIBCUDACXX_EXECUTOR
  "None"
  CACHE STRING
  "Executor to use when running tests."
)

set(
  LIBCUDACXX_TEST_TIMEOUT
  "200"
  CACHE STRING
  "Enable test timeouts (Default = 200, Off = 0)"
)

set(
  AUTO_GEN_COMMENT
  "## Autogenerated by libcudacxx configuration.\n# Do not edit!"
)

set(LIBCUDACXX_TEST_STANDARD_VER "c++${CMAKE_CUDA_STANDARD}")

set(lit_site_cfg_path "${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg")
configure_lit_site_cfg(
  "${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in"
  "${lit_site_cfg_path}"
)

add_lit_testsuite(check-cudacxx
  "Running libcu++ tests"
  "${CMAKE_CURRENT_BINARY_DIR}"
)

find_program(libcudacxx_LIT lit REQUIRED)

set(
  libcudacxx_LIT_FLAGS
  ""
  CACHE STRING
  "Semi-colon separated list of flags passed to the invocation of lit."
)
message(STATUS "libcudacxx_LIT_FLAGS: ${libcudacxx_LIT_FLAGS}")

if (NOT LIBCUDACXX_TEST_WITH_NVRTC)
  # Build but don't run the tests. Used by CI to pre-seed sccache for the test machines.
  # Only executed if explicitly requested.
  add_custom_target(
    libcudacxx.test.lit.precompile
    # HACK: There is no way to tell CMake/ninja to always build a target serially,
    # so we make this target depend on all other libcudacxx targets to avoid oversubscribing
    # the build machine.
    # FIXME: This has nasty side effects:
    # - It's fragile and must be updated every time we add new targets to libcudacxx
    # - It oversubs `-dev` presets that configure libcudacxx alongside other CCCL projects
    # - It makes it impossible to just build this target alone since it brings in the world
    # See related issue https://github.com/NVIDIA/cccl/issues/6163.
    DEPENDS
      libcudacxx.test.public_headers
      libcudacxx.test.internal_headers
      libcudacxx.test.public_headers_host_only
      libcudacxx.test.c2h_all
    # gersemi: off
    COMMAND
      "${CMAKE_COMMAND}" -E env "LIBCUDACXX_SITE_CONFIG=${lit_site_cfg_path}"
        "${libcudacxx_LIT}"
          -vv --no-progress-bar --time-tests
          ${libcudacxx_LIT_FLAGS}
          "-Dexecutor=\"NoopExecutor()\""
          "${libcudacxx_SOURCE_DIR}/test/libcudacxx"
    # gersemi: on
    USES_TERMINAL
  )
endif()

# Restricted to avoid oversubscribing the GPU:
set(
  libcudacxx_LIT_PARALLEL_LEVEL
  8
  CACHE STRING
  "Parallelism used to run libcudacxx's lit test suite."
)

add_test(
  NAME libcudacxx.test.lit
  # gersemi: off
  COMMAND
    "${CMAKE_COMMAND}" -E env "LIBCUDACXX_SITE_CONFIG=${lit_site_cfg_path}"
      "${libcudacxx_LIT}"
        -vv --no-progress-bar --time-tests
        ${libcudacxx_LIT_FLAGS}
        -j "${libcudacxx_LIT_PARALLEL_LEVEL}"
        "${libcudacxx_SOURCE_DIR}/test/libcudacxx"
    # gersemi: on
    USES_TERMINAL
)

set_tests_properties(
  libcudacxx.test.lit
  PROPERTIES
    # 6hr to match CI timeout
    TIMEOUT 21600
    RUN_SERIAL TRUE
)
