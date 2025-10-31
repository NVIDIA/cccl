find_package(CUDAToolkit REQUIRED)

if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL 12.6)
  add_subdirectory(legacy_interact)
endif()
