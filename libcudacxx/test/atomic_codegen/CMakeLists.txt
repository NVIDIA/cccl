add_custom_target(libcudacxx.test.atomics.ptx)

find_program(filecheck "FileCheck")

if (filecheck)
  message("-- ${filecheck} found... building atomic codegen tests")
else()
  return()
endif()

find_program(cuobjdump "cuobjdump" REQUIRED)
find_program(bash "bash" REQUIRED)

set(libcudacxx_atomic_codegen_tests)
if (NOT "NVHPC" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
  file(GLOB libcudacxx_atomic_codegen_tests "*.cu")
endif()

# For every atomic API compile the TU and check if the SASS/PTX matches the expected result
foreach(test_path IN LISTS libcudacxx_atomic_codegen_tests)
    cmake_path(GET test_path FILENAME test_file)
    cmake_path(REMOVE_EXTENSION test_file LAST_ONLY OUTPUT_VARIABLE test_name)

    add_library(
        atomic_codegen_${test_name}
        STATIC ${test_path}
    )

    set_target_properties(
        atomic_codegen_${test_name}
        PROPERTIES
          CUDA_ARCHITECTURES "80"
          COMPILE_DEFINITIONS "_LIBCUDACXX_ATOMIC_UNSAFE_AUTOMATIC_STORAGE=1"
    )

    target_compile_options(atomic_codegen_${test_name} PRIVATE "-Wno-comment")

    ## Important for testing the local headers
    target_include_directories(atomic_codegen_${test_name} PRIVATE "${libcudacxx_SOURCE_DIR}/include")
    add_dependencies(libcudacxx.test.atomics.ptx atomic_codegen_${test_name})

    # Add output path to object directory
    add_custom_command(
        TARGET libcudacxx.test.atomics.ptx
        POST_BUILD
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/dump_and_check.bash $<TARGET_FILE:atomic_codegen_${test_name}> ${test_path} SM8X
    )
endforeach()
