cmake_minimum_required(VERSION 3.8)

project(getsm CUDA)

add_executable(getsm
    main.cu)

add_custom_target(
    get-sm ALL
    COMMAND getsm
)
