## Codegen adds the following build targets
#    libcudacxx.atomics.codegen
#    libcudacxx.atomics.codegen.execute
#    libcudacxx.atomics.codegen.install
## Test targets:
#    libcudacxx.atomics.codegen.diff

add_custom_target(libcudacxx.atomics.codegen)

add_executable(
    codegen
    EXCLUDE_FROM_ALL
    codegen.cpp
)

target_compile_features(
    codegen PRIVATE cxx_std_14
)

add_dependencies(libcudacxx.atomics.codegen codegen)

set(atomic_generated_output "${libcudacxx_BINARY_DIR}/codegen/cuda_ptx_generated.h")
set(atomic_install_location "${libcudacxx_SOURCE_DIR}/include/cuda/std/__atomic/functions")

add_custom_target(
    libcudacxx.atomics.codegen.execute
    COMMAND codegen
    BYPRODUCTS "${atomic_generated_output}"
)

add_dependencies(libcudacxx.atomics.codegen libcudacxx.atomics.codegen.execute)

add_custom_target(
    libcudacxx.atomics.codegen.install
    COMMAND ${CMAKE_COMMAND} -E copy "${atomic_generated_output}" "${atomic_install_location}/cuda_ptx_generated.h"
    BYPRODUCTS "${atomic_install_location}/cuda_ptx_generated.h"
)

add_dependencies(libcudacxx.atomics.codegen.install libcudacxx.atomics.codegen.execute)

add_test(
    NAME libcudacxx.atomics.codegen.diff
    COMMAND ${CMAKE_COMMAND} -E compare_files "${atomic_install_location}/cuda_ptx_generated.h" "${atomic_generated_output}"
)
