## Codegen adds the following build targets
#    libcudacxx.atomics.codegen
#    libcudacxx.atomics.codegen.install
## Test targets:
#    libcudacxx.test.atomics.codegen.diff

cccl_get_fmt()

add_executable(
    codegen
    EXCLUDE_FROM_ALL
    codegen.cpp
)

target_link_libraries(codegen PRIVATE fmt)

set_property(TARGET codegen PROPERTY CXX_STANDARD 17)

set(atomic_generated_output "${libcudacxx_BINARY_DIR}/codegen/cuda_ptx_generated.h")
set(atomic_install_location "${libcudacxx_SOURCE_DIR}/include/cuda/std/__atomic/functions")

add_custom_target(
    libcudacxx.atomics.codegen
    COMMAND codegen "${atomic_generated_output}"
    BYPRODUCTS "${atomic_generated_output}"
)

add_custom_target(
    libcudacxx.atomics.codegen.install
    COMMAND ${CMAKE_COMMAND} -E copy "${atomic_generated_output}" "${atomic_install_location}/cuda_ptx_generated.h"
    DEPENDS libcudacxx.atomics.codegen
    BYPRODUCTS "${atomic_install_location}/cuda_ptx_generated.h"
)

add_test(
    NAME libcudacxx.test.atomics.codegen.diff
    COMMAND ${CMAKE_COMMAND} -E compare_files "${atomic_install_location}/cuda_ptx_generated.h" "${atomic_generated_output}"
)

set_tests_properties(
    libcudacxx.test.atomics.codegen.diff
    PROPERTIES REQUIRED_FILES "${atomic_generated_output}"
)
