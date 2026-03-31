# This unit just checks that nv/target can be used with the specified dialects.

set(nvtarget_dialects 11 14 17 20)

# cudafe++ 12.9 has problem compiling some parts of <type_traits> when using libstdc++ 14, so let's just filter out
# dialect version 11.
if (
  "${CMAKE_CUDA_COMPILER_ID}" STREQUAL "NVIDIA"
  AND "${CMAKE_CUDA_COMPILER_VERSION}" VERSION_GREATER_EQUAL "12.9"
  AND "${CMAKE_CUDA_COMPILER_VERSION}" VERSION_LESS "13.0"
  AND "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU"
  AND "${CMAKE_CXX_COMPILER_VERSION}" VERSION_GREATER_EQUAL "14.0"
  AND "${CMAKE_CXX_COMPILER_VERSION}" VERSION_LESS "15.0"
)
  list(FILTER nvtarget_dialects EXCLUDE REGEX "^11$")
endif()

set_directory_properties(
  PROPERTIES INCLUDE_DIRECTORIES "${libcudacxx_SOURCE_DIR}/include"
)

foreach (dialect ${nvtarget_dialects})
  if ("cxx_std_${dialect}" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    set(target_name "libcudacxx.test.nvtarget.dialect.cpp${dialect}")
    add_library(${target_name} OBJECT nvtargettest.cpp)
    # This is required to override CMAKE_CXX_STANDARD - target_compile_features is insufficient.
    set_target_properties(
      ${target_name}
      PROPERTIES CXX_STANDARD "${dialect}" CXX_STANDARD_REQUIRED ON
    )
    add_dependencies(libcudacxx.test.nvtarget ${target_name})
  endif()

  if (
    "cuda_std_${dialect}" IN_LIST CMAKE_CUDA_COMPILE_FEATURES
    AND "cxx_std_${dialect}" IN_LIST CMAKE_CXX_COMPILE_FEATURES
  )
    set(target_name "libcudacxx.test.nvtarget.dialect.cuda${dialect}")
    add_library(${target_name} OBJECT nvtargettest.cu)
    # This is required to override CMAKE_CXX_STANDARD - target_compile_features is insufficient.
    set_target_properties(
      ${target_name}
      PROPERTIES CUDA_STANDARD "${dialect}" CUDA_STANDARD_REQUIRED ON
    )
    add_dependencies(libcudacxx.test.nvtarget ${target_name})
  endif()
endforeach()
