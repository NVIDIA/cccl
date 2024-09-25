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

function(cccl_build_compiler_interface interface_target cuda_compile_options cxx_compile_options compile_defs)
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
  list(APPEND cuda_compile_options "-Xcudafe=--promote_warnings")
  list(APPEND cuda_compile_options "-Wno-deprecated-gpu-targets")

  # Ensure that we build our tests without treating ourself as system header
  list(APPEND cxx_compile_definitions "_CCCL_NO_SYSTEM_HEADER")

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
    if (CMAKE_BUILD_TYPE STREQUAL "Release")
      append_option_if_available("/WX" cxx_compile_options)
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

  else()
    list(APPEND cuda_compile_options "-Wreorder")

    append_option_if_available("-Werror" cxx_compile_options)
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

  if ("Intel" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
    # Do not flush denormal floats to zero
    append_option_if_available("-no-ftz" cxx_compile_options)
    # Disable warning that inlining is inhibited by compiler thresholds.
    append_option_if_available("-diag-disable=11074" cxx_compile_options)
    append_option_if_available("-diag-disable=11076" cxx_compile_options)
    # Disable warning about deprecated classic compiler
    append_option_if_available("-diag-disable=10441" cxx_compile_options)
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

  if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # C4127: conditional expression is constant
    # Disable this MSVC warning for C++11/C++14. In C++17+, we can use
    # _CCCL_IF_CONSTEXPR to address these warnings.
    target_compile_options(cccl.compiler_interface_cpp11 INTERFACE
      $<$<COMPILE_LANGUAGE:CXX>:/wd4127>
      $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-Xcompiler=/wd4127>
    )
    target_compile_options(cccl.compiler_interface_cpp14 INTERFACE
      $<$<COMPILE_LANGUAGE:CXX>:/wd4127>
      $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-Xcompiler=/wd4127>
    )
  endif()

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
