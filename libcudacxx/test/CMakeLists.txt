# Libcudacxx's main lit tests
add_subdirectory(libcudacxx)

# Set appropriate warning levels for MSVC/sane
if("${CMAKE_CUDA_COMPILER_ID}" STREQUAL "NVIDIA")
  # CUDA 11.5 and down do not support '-use-local-env'
  if(MSVC)
    set(headertest_warning_levels_device -Xcompiler=/W4 -Xcompiler=/WX)
    if ("${CMAKE_CUDA_COMPILER_VERSION}" GREATER_EQUAL "11.6.0")
      list(APPEND headertest_warning_levels_device --use-local-env)
    endif()
  else()
    set(headertest_warning_levels_device -Wall -Werror all-warnings)
  endif()
# Set warnings for Clang as device compiler
elseif("${CMAKE_CUDA_COMPILER_ID}" STREQUAL "Clang")
  set(headertest_warning_levels_device -Wall -Werror -Wno-unknown-cuda-version -Xclang=-fcuda-allow-variadic-functions)
# If the CMAKE_CUDA_COMPILER is unknown, try to use gcc style warnings
else()
  set(headertest_warning_levels_device -Wall -Werror)
endif()

# Set raw host/device warnings
if(MSVC)
  set(headertest_warning_levels_host /W4 /WX)
else()
  set(headertest_warning_levels_host -Wall -Werror)
endif()

# Libcudacxx auto-generated internal header tests
add_subdirectory(internal_headers)

# Libcudacxx auto-generated public header tests
add_subdirectory(public_headers)

# Libcudacxx auto-generated public header as std tests
add_subdirectory(public_headers_host_only)

# Enable building the nvrtcc project if NVRTC is enabled
if (LIBCUDACXX_TEST_WITH_NVRTC)
  add_subdirectory(utils/nvidia/nvrtc)
endif()

find_program(filecheck "FileCheck")

if (filecheck)
  message("-- ${filecheck} found... building atomic codegen tests")
  add_subdirectory(atomic_codegen)
endif()
