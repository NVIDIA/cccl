# Libcudacxx's main lit tests
add_subdirectory(libcudacxx)

# Libcudacxx auto-generated include/smoke tests
add_subdirectory(smoke)

# Enable building the nvrtcc project if NVRTC is enabled
if (LIBCUDACXX_TEST_WITH_NVRTC)
  add_subdirectory(utils/nvidia/nvrtc)
endif()
