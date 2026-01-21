# This unit just checks that nv/target can be used with the specified dialects.

set(
  nvtarget_dialects
  11
  14
  17
  20
)

set_directory_properties(
  PROPERTIES
  INCLUDE_DIRECTORIES "${libcudacxx_SOURCE_DIR}/include"
)

add_custom_target(libcudacxx.test.nvtarget)

foreach(dialect ${nvtarget_dialects})
  if ("cxx_std_${dialect}" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    add_library("nvtarget_cpp${dialect}" OBJECT nvtargettest.cpp)
    # This is required to override CMAKE_CXX_STANDARD - target_compile_features is insufficient.
    set_target_properties(
      "nvtarget_cpp${dialect}" PROPERTIES
      CXX_STANDARD "${dialect}"
      CXX_STANDARD_REQUIRED ON
    )
  endif()

  if ("cuda_std_${dialect}" IN_LIST CMAKE_CUDA_COMPILE_FEATURES)
    add_library("nvtarget_cuda${dialect}" OBJECT nvtargettest.cu)
    # This is required to override CMAKE_CXX_STANDARD - target_compile_features is insufficient.
    set_target_properties(
      "nvtarget_cuda${dialect}" PROPERTIES
      CUDA_STANDARD "${dialect}"
      CUDA_STANDARD_REQUIRED ON
    )
  endif()

  add_dependencies(libcudacxx.test.nvtarget "nvtarget_cpp${dialect}" "nvtarget_cuda${dialect}")
endforeach()
