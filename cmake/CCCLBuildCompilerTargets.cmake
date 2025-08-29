# This file defines the `cccl_build_compiler_targets()` function, which
# creates the following interface targets:
#
# cccl.compiler_interface
# - Interface target providing compiler-specific options needed to build
#   CCCL's tests, examples, etc. This includes warning flags and the like.
#
# cccl.compiler_interface_cppXX
# - Interface targets providing compiler-specific options that should only be
#   applied to certain dialects of C++. Includes `compiler_interface`, and will
#   be defined for each supported dialect.
#
# cccl.silence_unreachable_code_warnings
# - Interface target that silences unreachable code warnings.
# - Used to selectively disable such warnings in unit tests caused by
#   unconditionally thrown exceptions.

set(CCCL_KNOWN_CXX_DIALECTS 11 14 17 20)

# sccache cannot handle the -Fd option generating pdb files
set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT Embedded)

option(CCCL_ENABLE_EXCEPTIONS "Enable exceptions within CCCL libraries." ON)
option(CCCL_ENABLE_RTTI "Enable RTTI within CCCL libraries." ON)
option(CCCL_ENABLE_WERROR "Treat warnings as errors for CCCL targets." ON)

function(cccl_build_compiler_interface interface_target cuda_compile_options cxx_compile_options compile_defs)
  # We test to see if C++ compiler options exist using try-compiles in the CXX lang, and then reuse those flags as
  # -Xcompiler flags for CUDA targets. This requires that the CXX compiler and CUDA_HOST compilers are the same when
  # using nvcc.
  if (CCCL_TOPLEVEL_PROJECT AND CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    set(cuda_host_matches_cxx_compiler FALSE)
    if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.31)
      set(host_info "${CMAKE_CUDA_HOST_COMPILER} (${CMAKE_CUDA_HOST_COMPILER_ID} ${CMAKE_CUDA_HOST_COMPILER_VERSION})")
      set(cxx_info "${CMAKE_CXX_COMPILER} (${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION})")
      if (CMAKE_CUDA_HOST_COMPILER_ID STREQUAL CMAKE_CXX_COMPILER_ID AND
          CMAKE_CUDA_HOST_COMPILER_VERSION VERSION_EQUAL CMAKE_CXX_COMPILER_VERSION)
        set(cuda_host_matches_cxx_compiler TRUE)
      endif()
    else() # CMake < 3.31 doesn't have the CMAKE_CUDA_HOST_COMPILER_ID/VERSION variables
      set(host_info "${CMAKE_CUDA_HOST_COMPILER}")
      set(cxx_info "${CMAKE_CXX_COMPILER}")
      if (CMAKE_CUDA_HOST_COMPILER STREQUAL CMAKE_CXX_COMPILER)
        set(cuda_host_matches_cxx_compiler TRUE)
      endif()
    endif()

    if (NOT cuda_host_matches_cxx_compiler)
      message(FATAL_ERROR
        "CCCL developer builds require that CMAKE_CUDA_HOST_COMPILER matches "
        "CMAKE_CXX_COMPILER when using nvcc:\n"
        "CMAKE_CUDA_COMPILER: ${CMAKE_CUDA_COMPILER}\n"
        "CMAKE_CUDA_HOST_COMPILER: ${host_info}\n"
        "CMAKE_CXX_COMPILER: ${cxx_info}\n"
        "Rerun cmake with \"-DCMAKE_CUDA_HOST_COMPILER=${CMAKE_CXX_COMPILER}\".\n"
        "Alternatively, configure the CUDAHOSTCXX and CXX environment variables to match.\n"
      )
    endif()
  endif()

  add_library(${interface_target} INTERFACE)

  foreach (cuda_option IN LISTS cuda_compile_options)
    target_compile_options(${interface_target} INTERFACE
      $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:${cuda_option}>
    )
  endforeach()

  foreach (cxx_option IN LISTS cxx_compile_options)
    target_compile_options(${interface_target} INTERFACE
      $<$<COMPILE_LANGUAGE:CXX>:${cxx_option}>
      $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-Xcompiler=${cxx_option}>
    )
  endforeach()

  target_compile_definitions(${interface_target} INTERFACE
    ${compile_defs}
  )
endfunction()

function(cccl_build_compiler_targets)
  set(cuda_compile_options)
  set(cxx_compile_options)
  set(cxx_compile_definitions)

  list(APPEND cuda_compile_options "-Xcudafe=--display_error_number")
  list(APPEND cuda_compile_options "-Wno-deprecated-gpu-targets")
  if (CCCL_ENABLE_WERROR)
    list(APPEND cuda_compile_options "-Xcudafe=--promote_warnings")
  endif()

  # Ensure that we build our tests without treating ourself as system header
  list(APPEND cxx_compile_definitions "_CCCL_NO_SYSTEM_HEADER")

  if (NOT CCCL_ENABLE_EXCEPTIONS)
    list(APPEND cxx_compile_definitions "CCCL_DISABLE_EXCEPTIONS")
  endif()

  if (NOT CCCL_ENABLE_RTTI)
    list(APPEND cxx_compile_definitions "CCCL_DISABLE_RTTI")
  endif()

  if ("MSVC" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
    list(APPEND cuda_compile_options "--use-local-env")
    list(APPEND cxx_compile_options "/bigobj")
    list(APPEND cxx_compile_definitions "_ENABLE_EXTENDED_ALIGNED_STORAGE")
    list(APPEND cxx_compile_definitions "NOMINMAX")

    append_option_if_available("/W4" cxx_compile_options)
    # Treat all warnings as errors. This is only supported on Release builds,
    # as `nv_exec_check_disable` doesn't seem to work with MSVC debug iterators
    # and spurious warnings are emitted.
    # See NVIDIA/thrust#1273, NVBug 3129879.
    if (CCCL_ENABLE_WERROR)
      if (CMAKE_BUILD_TYPE STREQUAL "Release")
        append_option_if_available("/WX" cxx_compile_options)
      endif()
    endif()

    # Suppress overly-pedantic/unavoidable warnings brought in with /W4:
    # C4324: structure was padded due to alignment specifier
    append_option_if_available("/wd4324" cxx_compile_options)
    # C4505: unreferenced local function has been removed
    # The CUDA `host_runtime.h` header emits this for
    # `__cudaUnregisterBinaryUtil`.
    append_option_if_available("/wd4505" cxx_compile_options)
    # C4706: assignment within conditional expression
    # MSVC doesn't provide an opt-out for this warning when the assignment is
    # intentional. Clang will warn for these, but suppresses the warning when
    # double-parentheses are used around the assignment. We'll let Clang catch
    # unintentional assignments and suppress all such warnings on MSVC.
    append_option_if_available("/wd4706" cxx_compile_options)

    # Disabled loss-of-data conversion warnings.
    # append_option_if_available("/wd4244" cxx_compile_options)

    # Disable warning about applying unary operator- to unsigned type.
    # append_option_if_available("/wd4146" cxx_compile_options)

    # MSVC STL assumes that `allocator_traits`'s allocator will use raw pointers,
    # and the `__DECLSPEC_ALLOCATOR` macro causes issues with thrust's universal
    # allocators:
    #   warning C4494: 'std::allocator_traits<_Alloc>::allocate' :
    #      Ignoring __declspec(allocator) because the function return type is not
    #      a pointer or reference
    # See https://github.com/microsoft/STL/issues/696
    append_option_if_available("/wd4494" cxx_compile_options)

    # We require the new preprocessor
    append_option_if_available("/Zc:preprocessor" cxx_compile_options)

  else()
    list(APPEND cuda_compile_options "-Wreorder")

    if (CCCL_ENABLE_WERROR)
      append_option_if_available("-Werror" cxx_compile_options)
    endif()

    append_option_if_available("-Wall" cxx_compile_options)
    append_option_if_available("-Wextra" cxx_compile_options)
    append_option_if_available("-Wreorder" cxx_compile_options)
    append_option_if_available("-Winit-self" cxx_compile_options)
    append_option_if_available("-Woverloaded-virtual" cxx_compile_options)
    append_option_if_available("-Wcast-qual" cxx_compile_options)
    append_option_if_available("-Wpointer-arith" cxx_compile_options)
    append_option_if_available("-Wunused-local-typedef" cxx_compile_options)
    append_option_if_available("-Wvla" cxx_compile_options)

    # Disable GNU extensions (flag is clang only)
    append_option_if_available("-Wgnu" cxx_compile_options)
    append_option_if_available("-Wno-gnu-line-marker" cxx_compile_options) # WAR 3916341
    # Calling a variadic macro with zero args is a GNU extension until C++20,
    # but the THRUST_PP_ARITY macro is used with zero args. Need to see if this
    # is a real problem worth fixing.
    append_option_if_available("-Wno-gnu-zero-variadic-macro-arguments" cxx_compile_options)

    # This complains about functions in CUDA system headers when used with nvcc.
    append_option_if_available("-Wno-unused-function" cxx_compile_options)
  endif()

  if ("GNU" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 7.3)
      # GCC 7.3 complains about name mangling changes due to `noexcept`
      # becoming part of the type system; we don't care.
      append_option_if_available("-Wno-noexcept-type" cxx_compile_options)
    endif()
  endif()

  cccl_build_compiler_interface(cccl.compiler_interface
    "${cuda_compile_options}"
    "${cxx_compile_options}"
    "${cxx_compile_definitions}"
  )

  # These targets are used for dialect-specific options:
  foreach (dialect IN LISTS CCCL_KNOWN_CXX_DIALECTS)
    add_library(cccl.compiler_interface_cpp${dialect} INTERFACE)
    target_link_libraries(cccl.compiler_interface_cpp${dialect} INTERFACE cccl.compiler_interface)
  endforeach()

  # Some of our unit tests unconditionally throw exceptions, and compilers will
  # detect that the following instructions are unreachable. This is intentional
  # and unavoidable in these cases. This target can be used to silence
  # unreachable code warnings.
  add_library(cccl.silence_unreachable_code_warnings INTERFACE)
  if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(cccl.silence_unreachable_code_warnings INTERFACE
      $<$<COMPILE_LANGUAGE:CXX>:/wd4702>
      $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-Xcompiler=/wd4702>
    )
  endif()
endfunction()
