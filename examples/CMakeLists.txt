cmake_minimum_required(VERSION 3.10)

enable_testing()

function(add_example target_name example_dir)
    add_subdirectory(${example_dir})

    # Add a test that will run the executable
    # Assumes that the executable target has the same name as the directory
    add_test(NAME ${target_name}
        COMMAND "${CMAKE_COMMAND}"
        "-DEXAMPLE_EXECUTABLE=$<TARGET_FILE:${target_name}>"
        -P "${CMAKE_SOURCE_DIR}/cmake/RunExample.cmake"
    )
endfunction()

file(GLOB children RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
foreach(child ${children})
    # Grab all subdirectories except build/ and cmake/
    if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${child} AND NOT ${child} STREQUAL "build" AND NOT ${child} STREQUAL "cmake")
        add_example(${child} ${child})
    endif()
endforeach()
