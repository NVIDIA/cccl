add_subdirectory(atomic_codegen)
add_subdirectory(cmake)
add_subdirectory(libcudacxx)

# Enable building the nvrtcc project if NVRTC is enabled
if (LIBCUDACXX_TEST_WITH_NVRTC)
  add_subdirectory(utils/nvidia/nvrtc)
endif()
