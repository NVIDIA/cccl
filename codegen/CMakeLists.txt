cmake_minimum_required(VERSION 3.8)

project(cuda-atomics-codegen CXX)

set(CMAKE_CXX_STANDARD 14)
add_executable(codegen
    codegen.cpp
)

add_custom_target(
    run ALL
    COMMAND codegen
    BYPRODUCS "${libcudacxx_BINARY_DIR}/__atomic_generated"
)

install(
    FILES "${libcudacxx_BINARY_DIR}/__atomic_generated"
    DESTINATION "${libcudacxx_SOURCE_DIR}/../include/cuda/std/detail/"
)
