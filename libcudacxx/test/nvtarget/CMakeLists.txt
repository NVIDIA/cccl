# This unit just checks that nv/target can be used with the specified dialects.

if (MSVC)
  set(
    nvtarget_dialects
    14
    17
    20
  )
else()
  set(
    nvtarget_dialects
    11
    14
    17
    20
  )
endif()

add_custom_target(libcudacxx.test.nvtarget)

foreach(dialect ${nvtarget_dialects})
  add_library(
    "nvtarget_cpp${dialect}"
    OBJECT
    nvtargettest.cpp
  )
  add_library(
    "nvtarget_cuda${dialect}"
    OBJECT
    nvtargettest.cu
  )

  target_include_directories(
    "nvtarget_cpp${dialect}"
    PRIVATE "${libcudacxx_SOURCE_DIR}/include"
  )
  target_include_directories(
    "nvtarget_cuda${dialect}"
    PRIVATE "${libcudacxx_SOURCE_DIR}/include"
  )

  set_target_properties(
    "nvtarget_cpp${dialect}"
    PROPERTIES
      COMPILE_OPTIONS "${headertest_warning_levels_host}"
      CXX_STANDARD "${dialect}"
      CXX_STANDARD_REQUIRED ON
  )
  set_target_properties(
    "nvtarget_cuda${dialect}"
    PROPERTIES
      COMPILE_OPTIONS "${headertest_warning_levels_device}"
      CUDA_STANDARD "${dialect}"
      CUDA_STANDARD_REQUIRED ON
  )

  add_dependencies(libcudacxx.test.nvtarget "nvtarget_cpp${dialect}" "nvtarget_cuda${dialect}")
endforeach()
