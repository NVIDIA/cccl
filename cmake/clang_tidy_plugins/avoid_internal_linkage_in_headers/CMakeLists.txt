list(APPEND CMAKE_MESSAGE_CONTEXT "avoid_internal_linkage_in_headers")

cccl_create_clang_tidy_plugin(
  cccl.tidy.avoid_internal_linkage_in_headers
  SOURCES avoid_internal_linkage_in_headers.cpp
)

add_subdirectory(test)
