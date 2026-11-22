list(APPEND CMAKE_MESSAGE_CONTEXT "hide_from_abi")

cccl_create_clang_tidy_plugin(hide_from_abi.cpp)

add_subdirectory(test)
