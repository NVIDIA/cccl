# Libcudacxx's main lit tests
add_subdirectory(libcudacxx)

# Set appropriate warning levels for MSVC/sane
if("${CMAKE_CUDA_COMPILER_ID}" STREQUAL "NVIDIA")
  if(MSVC)
    set(headertest_warning_levels_device -use-local-env -Xcompiler=/W4 -Xcompiler=/WX)
  else()
    set(headertest_warning_levels_device -Wall -Werror)
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

# Libcudacxx auto-generated include/smoke tests
add_subdirectory(public_headers)

# Enable building the nvrtcc project if NVRTC is enabled
if (LIBCUDACXX_TEST_WITH_NVRTC)
  add_subdirectory(utils/nvidia/nvrtc)
endif()
