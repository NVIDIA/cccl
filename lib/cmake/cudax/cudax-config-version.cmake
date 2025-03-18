set(cudax_VERSION_MAJOR 3)
set(cudax_VERSION_MINOR 1)
set(cudax_VERSION_PATCH 0)
set(cudax_VERSION_TWEAK 0)

set(cudax_VERSION "${cudax_VERSION_MAJOR}.${cudax_VERSION_MINOR}.${cudax_VERSION_PATCH}.${cudax_VERSION_TWEAK}")

set(PACKAGE_VERSION ${cudax_VERSION})
set(PACKAGE_VERSION_COMPATIBLE FALSE)
set(PACKAGE_VERSION_EXACT FALSE)
set(PACKAGE_VERSION_UNSUITABLE FALSE)

# Semantic versioning:
if(PACKAGE_VERSION VERSION_GREATER_EQUAL PACKAGE_FIND_VERSION)
  if(cudax_VERSION_MAJOR VERSION_EQUAL PACKAGE_FIND_VERSION_MAJOR)
    set(PACKAGE_VERSION_COMPATIBLE TRUE)
  endif()

  if(PACKAGE_FIND_VERSION VERSION_EQUAL PACKAGE_VERSION)
    set(PACKAGE_VERSION_EXACT TRUE)
  endif()
endif()
