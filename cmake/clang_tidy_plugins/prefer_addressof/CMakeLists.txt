list(APPEND CMAKE_MESSAGE_CONTEXT "prefer_addressof")

cccl_create_clang_tidy_plugin(prefer_addressof.cpp)

add_subdirectory(test)
