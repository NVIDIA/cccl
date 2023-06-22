# For standalone static library builds, default the C and ASM compilers to
# the C++ compiler.
if (LIBCUDACXX_ENABLE_STATIC_LIBRARY)
  if ("${CMAKE_C_COMPILER}" STREQUAL "")
    # Force the C compiler to avoid CMake complaining about a C++ compiler
    # being used for C.
    set(CMAKE_C_COMPILER_FORCED ON)
    set(CMAKE_C_COMPILER "${CMAKE_CXX_COMPILER}")
  endif ()
  if ("${CMAKE_ASM_COMPILER}" STREQUAL "")
    set(CMAKE_ASM_COMPILER "${CMAKE_CXX_COMPILER}")
  endif ()
endif ()
