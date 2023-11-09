add_custom_target(libcudacxx_tu_tests)

set(cpp_std_versions 11 14 17 20)

set(cpp_11_exclusions "cuda/annotated_ptr" "cuda/std/mdspan")
set(cpp_14_exclusions "cuda/annotated_ptr" "cuda/std/mdspan")
set(cpp_17_exclusions "cuda/annotated_ptr")
set(cpp_20_exclusions "cuda/annotated_ptr")

function(libcudacxx_add_standalone_header_test test_file_output_var header_under_test version)
    # ex: build/.../header_tests/cuda/std/version.cpp
    set(test_cpp "${CMAKE_CURRENT_BINARY_DIR}/header_tests/${header_under_test}.cpp")

    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/header_test.cpp.in"
        "${test_cpp}"
        )

    set(${test_file_output_var} ${test_cpp} PARENT_SCOPE)
    # ex: cuda/std/version -> cuda_std_version

    string(REPLACE "/" "_" executable_name "${header_under_test}-${version}")
    list(FIND cpp_${version}_exclusions ${header_under_test} header_excluded)

    if ("${header_excluded}" EQUAL "-1")
        add_library(${executable_name} OBJECT ${test_cpp})
        target_include_directories(${executable_name} PRIVATE "${libcudacxx_SOURCE_DIR}/include")
        target_compile_options(${executable_name} PRIVATE
        $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
        -Wall -Werror>
        $<$<CXX_COMPILER_ID:MSVC>:
        /W4 /WX>)
        set_target_properties(${executable_name} PROPERTIES CXX_STANDARD ${version})
        set_target_properties(${executable_name} PROPERTIES CXX_STANDARD_REQUIRED ON)
        set_target_properties(${executable_name} PROPERTIES CXX_EXTENSIONS OFF)
        add_dependencies(libcudacxx_tu_tests ${executable_name})
    endif()

endfunction()

# Don'libcudacxx generate CUDA targets, they fail currently.
file(GLOB cuda_headers     LIST_DIRECTORIES false RELATIVE "${libcudacxx_SOURCE_DIR}/include" "${libcudacxx_SOURCE_DIR}/include/cuda/*")
file(GLOB cuda_std_headers LIST_DIRECTORIES false RELATIVE "${libcudacxx_SOURCE_DIR}/include" "${libcudacxx_SOURCE_DIR}/include/cuda/std/*")

foreach(version IN LISTS cpp_std_versions)
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
        set(cxx_version "/std:c++${version}")
    else()
        set(cxx_version "-std=c++${version}")
    endif()

    try_compile(
        dialect_supported
        ${CMAKE_BINARY_DIR}/dialect_check_${version} "${CMAKE_CURRENT_SOURCE_DIR}/detect_dialect.cpp"
        COMPILE_DEFINITIONS ${cxx_version}
    )

    if (dialect_supported)
        foreach(header IN LISTS cuda_headers cuda_std_headers)
            libcudacxx_add_standalone_header_test(test_file ${header} ${version})
        endforeach()
    endif()
endforeach()
