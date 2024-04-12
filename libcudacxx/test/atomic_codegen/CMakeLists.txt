# For every atomic API compile the TU and check if the SASS matches the expected result
add_custom_target(libcudacxx.test.atomic_codegen)

file(GLOB libcudacxx_atomic_codegen_tests "*.cu")

add_library(
    atomic_codegen_lib
    STATIC ${libcudacxx_atomic_codegen_tests}
)

## Important for testing the local headers
target_include_directories(atomic_codegen_lib PRIVATE "${libcudacxx_SOURCE_DIR}/include")

add_dependencies(libcudacxx.test.atomic_codegen atomic_codegen_lib)

find_program(filecheck "FileCheck" REQUIRED)
find_program(cuobjdump "cuobjdump" REQUIRED)
find_program(bash "bash" REQUIRED)

foreach(test_path IN LISTS libcudacxx_atomic_codegen_tests)
    cmake_path(GET test_path FILENAME test_file)
    cmake_path(REMOVE_EXTENSION test_file LAST_ONLY OUTPUT_VARIABLE test_name)

    # Add output path to object directory

    add_custom_command(
        TARGET libcudacxx.test.atomic_codegen
        POST_BUILD
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/dump_and_check.bash $<TARGET_FILE:atomic_codegen_lib> ${test_path} SM8X
    )
endforeach()
