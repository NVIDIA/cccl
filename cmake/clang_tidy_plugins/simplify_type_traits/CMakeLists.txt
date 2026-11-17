list(APPEND CMAKE_MESSAGE_CONTEXT "simplify_type_traits")

cccl_create_clang_tidy_plugin(simplify_type_traits.cpp)

add_subdirectory(test)
