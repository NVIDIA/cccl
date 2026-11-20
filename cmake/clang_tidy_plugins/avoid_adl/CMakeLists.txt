list(APPEND CMAKE_MESSAGE_CONTEXT "avoid_adl")

cccl_create_clang_tidy_plugin(avoid_adl.cpp)

add_subdirectory(test)
