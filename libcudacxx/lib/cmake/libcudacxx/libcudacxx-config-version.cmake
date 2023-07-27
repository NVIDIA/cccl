# Parse version information from version header:
include("${CMAKE_CURRENT_LIST_DIR}/libcudacxx-header-search.cmake")

file(READ "${_libcudacxx_VERSION_INCLUDE_DIR}/cuda/std/detail/__config"
  libcudacxx_VERSION_HEADER
)

string(REGEX MATCH
  "#define[ \t]+_LIBCUDACXX_CUDA_API_VERSION[ \t]+([0-9]+)" unused_var
  "${libcudacxx_VERSION_HEADER}"
)

set(libcudacxx_VERSION_FLAT ${CMAKE_MATCH_1})
math(EXPR libcudacxx_VERSION_MAJOR "${libcudacxx_VERSION_FLAT} / 1000000")
math(EXPR libcudacxx_VERSION_MINOR "(${libcudacxx_VERSION_FLAT} / 1000) % 1000")
math(EXPR libcudacxx_VERSION_PATCH "${libcudacxx_VERSION_FLAT} % 1000")
set(libcudacxx_VERSION_TWEAK 0)

set(libcudacxx_VERSION
  "${libcudacxx_VERSION_MAJOR}.${libcudacxx_VERSION_MINOR}.${libcudacxx_VERSION_PATCH}.${libcudacxx_VERSION_TWEAK}"
)

set(PACKAGE_VERSION ${libcudacxx_VERSION})
set(PACKAGE_VERSION_COMPATIBLE FALSE)
set(PACKAGE_VERSION_EXACT FALSE)
set(PACKAGE_VERSION_UNSUITABLE FALSE)

if(PACKAGE_VERSION VERSION_GREATER_EQUAL PACKAGE_FIND_VERSION)
  # Semantic version check:
  if(libcudacxx_VERSION_MAJOR VERSION_EQUAL PACKAGE_FIND_VERSION_MAJOR)
    set(PACKAGE_VERSION_COMPATIBLE TRUE)
  endif()

  # Special case: Major version 1->2 was bumped to sync with other CCCL
  # libraries. There was no break, and requests for 1 are compatible with 2.
  if(PACKAGE_FIND_VERSION VERSION_EQUAL 1 AND
     libcudacxx_VERSION_MAJOR VERSION_EQUAL 2)
    set(PACKAGE_VERSION_COMPATIBLE TRUE)
  endif()

    if(PACKAGE_FIND_VERSION VERSION_EQUAL PACKAGE_VERSION)
    set(PACKAGE_VERSION_EXACT TRUE)
  endif()
endif()
