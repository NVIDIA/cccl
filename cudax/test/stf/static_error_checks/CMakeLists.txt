set(STATIC_ASSERT_TESTS
    gnu_launch.cpp
    gnu_parallel_for.cpp
    static.cu
    task_prototype.cu
    stream_task_prototype.cu
    graph_task_prototype.cu
)

get_property(INCLUDES DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
list(TRANSFORM INCLUDES PREPEND "-I")
string(JOIN " " INCLUDES ${INCLUDES})

add_custom_target(static_assert)

foreach(SRC ${STATIC_ASSERT_TESTS})
    # Specify the executable name and the source files it should compile
    get_filename_component(TARGET ${SRC} NAME_WE)
    set(TARGET "static_error_checks.${TARGET}")

    set(cmd "${CMAKE_CUDA_COMPILER} --forward-unknown-to-host-compiler -c ${CMAKE_CURRENT_SOURCE_DIR}/${SRC} ${INCLUDES} ${CMAKE_CUDA_FLAGS} -o /dev/null")

    add_custom_target(${TARGET}
        COMMAND bash -c "${cmd} 2>&1 | grep -q 'static assertion';\
            R=(\"\${PIPESTATUS[@]}\");\
            if [ \${R[0]} = 0 ]; then\
                echo 'Compilation should fail for target \"${target}\". To repro, run: ${cmd}' 1>&2;\
                false;\
            elif [ \${R[1]} != 0 ]; then\
                echo 'Static assertion did not fail for target \"${target}\". To repro, run: ${cmd}' 1>&2;\
                false;\
            fi"
        VERBATIM
    )
    add_dependencies(static_assert ${TARGET})
endforeach()

