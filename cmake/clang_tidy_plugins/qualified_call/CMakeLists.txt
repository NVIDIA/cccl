list(APPEND CMAKE_MESSAGE_CONTEXT "qualified_call")

cccl_create_clang_tidy_plugin(qualified_call.cpp)

add_subdirectory(test)
