list(APPEND CMAKE_MESSAGE_CONTEXT "nodebug_on_aliases")

cccl_create_clang_tidy_plugin(nodebug_on_aliases.cpp)

add_subdirectory(test)
