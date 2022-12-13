function(libcudacxx_add_standalone_header_test test_file_output_var header_under_test)
    # ex: build/.../header_tests/cuda/std/version.cpp
    set(test_cpp "${CMAKE_CURRENT_BINARY_DIR}/header_tests/${header_under_test}.cpp")

    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/header_test.cpp.in"
        "${test_cpp}"
        )

    set(${test_file_output_var} ${test_cpp} PARENT_SCOPE)
    # ex: cuda/std/version -> cuda_std_version
    string(REPLACE "/" "_" executable_name ${header_under_test})
    add_executable(${executable_name} ${test_cpp})
    target_include_directories(${executable_name} PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_compile_options(${executable_name} PRIVATE
        $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
            -Wall -Werror>
        $<$<CXX_COMPILER_ID:MSVC>:
            /W4 /WX>)
    set_target_properties(${executable_name} PROPERTIES CXX_STANDARD 11)
endfunction()

# Don't generate CUDA targets, they fail currently.
# file(GLOB cuda_headers LIST_DIRECTORIES false RELATIVE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/include/cuda/*)
file(GLOB cuda_std_headers LIST_DIRECTORIES false RELATIVE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/include/cuda/std/*)

foreach(header IN LISTS cuda_headers cuda_std_headers)
    libcudacxx_add_standalone_header_test(test_file ${header})
    message(STATUS "Detected ${header} ... Writing ${test_file}")
endforeach()
