#
# This file defines the `CudaNext_build_compiler_targets()` function, which
# creates the following interface targets:
#
# CudaNext.compiler_interface
# - Interface target providing compiler-specific options needed to build
#   CudaNext's tests, examples, etc.

include("${CudaNext_SOURCE_DIR}/cmake/AppendOptionIfAvailable.cmake")

function(CudaNext_build_compiler_targets)
  set(cxx_compile_definitions)
  set(cxx_compile_options)
  set(cuda_compile_options)

  if ("MSVC" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
    # sccache cannot handle the -Fd option generating pdb files
    set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT Embedded)

    append_option_if_available("--use-local-env" cuda_compile_options)

    append_option_if_available("/W4" cxx_compile_options)

    append_option_if_available("/WX" cxx_compile_options)

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

    # CudaNext requires dim3 to be usable from a constexpr context, and the CUDART headers require
    # __cplusplus to be defined for this to work:
    append_option_if_available("/Zc:__cplusplus" cxx_compile_options)
  else()
    append_option_if_available("-Wreorder" cuda_compile_options)
    append_option_if_available("-Wno_unknown-cuda-version" cuda_compile_options)
    append_option_if_available("-Xclang=-fcuda-allow-variadic-functions" cuda_compile_options)

    append_option_if_available("-Werror" cxx_compile_options)
    append_option_if_available("-Wall" cxx_compile_options)
    append_option_if_available("-Wextra" cxx_compile_options)
    append_option_if_available("-Winit-self" cxx_compile_options)
    append_option_if_available("-Woverloaded-virtual" cxx_compile_options)
    append_option_if_available("-Wcast-qual" cxx_compile_options)
    append_option_if_available("-Wpointer-arith" cxx_compile_options)
    append_option_if_available("-Wunused-local-typedef" cxx_compile_options)
    append_option_if_available("-Wvla" cxx_compile_options)

    # Disable GNU extensions (flag is clang only)
    append_option_if_available("-Wgnu" cxx_compile_options)
    append_option_if_available("-Wno-gnu-line-marker" cxx_compile_options) # WAR 3916341

    # This complains about functions in CUDA system headers when used with nvcc.
    append_option_if_available("-Wno-unused-function" cxx_compile_options)

    # GCC 7.3 complains about name mangling changes due to `noexcept`
    append_option_if_available("-Wno-noexcept-type" cxx_compile_options)
  endif()

  if ("Clang" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
    option(CudaNext_ENABLE_CT_PROFILING "Enable compilation time profiling" OFF)
    if (CudaNext_ENABLE_CT_PROFILING)
      append_option_if_available("-ftime-trace" cxx_compile_options)
    endif()
  endif()

  add_library(CudaNext.compiler_interface INTERFACE)

  foreach (cxx_option IN LISTS cxx_compile_options)
    target_compile_options(CudaNext.compiler_interface INTERFACE
      $<$<COMPILE_LANGUAGE:CXX>:${cxx_option}>
      $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-Xcompiler=${cxx_option}>
    )
  endforeach()

  foreach (cuda_option IN LISTS cuda_compile_options)
    target_compile_options(CudaNext.compiler_interface INTERFACE
      $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:${cuda_option}>
    )
  endforeach()

  # Add these for both CUDA and CXX targets:
  target_compile_definitions(CudaNext.compiler_interface INTERFACE
    ${cxx_compile_definitions}
  )

  # Promote warnings and display diagnostic numbers for nvcc:
  target_compile_options(CudaNext.compiler_interface INTERFACE
    # If using CUDA w/ NVCC...
    # Display diagnostic numbers.
    $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-Xcudafe=--display_error_number>
    # Promote warnings.
    $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-Xcudafe=--promote_warnings>
    # Don't complain about deprecated GPU targets.
    $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-Wno-deprecated-gpu-targets>
  )
endfunction()
