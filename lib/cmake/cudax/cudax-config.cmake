if (TARGET cudax::cudax)
  return()
endif()

#
# Setup dependencies
#

set(cudax_quiet_flag "")
if (${CMAKE_FIND_PACKAGE_NAME}_FIND_QUIETLY)
  set(cudax_quiet_flag "QUIET")
endif()

set(cudax_cmake_dir "${CMAKE_CURRENT_LIST_DIR}")
set(cudax_prefix_dir "${cudax_cmake_dir}/../../..")
find_path(cudax_include_dir "cuda/experimental/version.cuh"
  REQUIRED
  NO_CACHE
  NO_CMAKE_FIND_ROOT_PATH # Don't allow CMake to re-root the search
  NO_DEFAULT_PATH # Only search explicit paths below:
  HINTS
    "${cudax_prefix_dir}/cudax/include" # Source
    "${cudax_prefix_dir}/include"       # Install
)

if (NOT TARGET cudax::libcudacxx)
  if (NOT TARGET libcudacxx::libcudacxx)
    # First do a non-required search for any co-packaged versions.
    # These are preferred.
    find_package(libcudacxx ${cudax_VERSION} CONFIG
      ${cudax_quiet_flag}
      NO_DEFAULT_PATH # Only check the explicit HINTS below:
      HINTS "${cudax_cmake_dir}/../libcudacxx"
    )

    # A second required search allows externally packaged to be used and fails if
    # no suitable package exists.
    find_package(libcudacxx ${cudax_VERSION} CONFIG
      REQUIRED
      ${cudax_quiet_flag}
    )
  endif()
  add_library(cudax::libcudacxx ALIAS libcudacxx::libcudacxx)
endif()

# Allow non-imported targets to be used. This treats cudax headers as non-system includes,
# exposing warnings in them. Used when building cudax tests.
if (cudax_NO_IMPORTED_TARGETS)
  # INTERFACE libraries cannot be namespaced; ALIAS libraries can.
  add_library(_cudax_cudax INTERFACE)
  add_library(cudax::cudax ALIAS _cudax_cudax)
  set(cudax_target_name _cudax_cudax)
else()
  add_library(cudax::cudax INTERFACE IMPORTED GLOBAL)
  set(cudax_target_name cudax::cudax)
endif()

target_compile_features(${cudax_target_name} INTERFACE cxx_std_17)
target_include_directories(${cudax_target_name} INTERFACE "${cudax_include_dir}")
target_link_libraries(${cudax_target_name} INTERFACE cudax::libcudacxx)

# Expose version info globally through cache variables:
set(cudax_VERSION       ${${CMAKE_FIND_PACKAGE_NAME}_VERSION}       CACHE INTERNAL "" FORCE)
set(cudax_VERSION_MAJOR ${${CMAKE_FIND_PACKAGE_NAME}_VERSION_MAJOR} CACHE INTERNAL "" FORCE)
set(cudax_VERSION_MINOR ${${CMAKE_FIND_PACKAGE_NAME}_VERSION_MINOR} CACHE INTERNAL "" FORCE)
set(cudax_VERSION_PATCH ${${CMAKE_FIND_PACKAGE_NAME}_VERSION_PATCH} CACHE INTERNAL "" FORCE)
set(cudax_VERSION_TWEAK ${${CMAKE_FIND_PACKAGE_NAME}_VERSION_TWEAK} CACHE INTERNAL "" FORCE)
set(cudax_VERSION_COUNT ${${CMAKE_FIND_PACKAGE_NAME}_VERSION_COUNT} CACHE INTERNAL "" FORCE)

# Uppercase for convenience:
set(CUDAX_VERSION       ${${CMAKE_FIND_PACKAGE_NAME}_VERSION}       CACHE INTERNAL "" FORCE)
set(CUDAX_VERSION_MAJOR ${${CMAKE_FIND_PACKAGE_NAME}_VERSION_MAJOR} CACHE INTERNAL "" FORCE)
set(CUDAX_VERSION_MINOR ${${CMAKE_FIND_PACKAGE_NAME}_VERSION_MINOR} CACHE INTERNAL "" FORCE)
set(CUDAX_VERSION_PATCH ${${CMAKE_FIND_PACKAGE_NAME}_VERSION_PATCH} CACHE INTERNAL "" FORCE)
set(CUDAX_VERSION_TWEAK ${${CMAKE_FIND_PACKAGE_NAME}_VERSION_TWEAK} CACHE INTERNAL "" FORCE)
set(CUDAX_VERSION_COUNT ${${CMAKE_FIND_PACKAGE_NAME}_VERSION_COUNT} CACHE INTERNAL "" FORCE)

include(FindPackageHandleStandardArgs)
if (NOT cudax_CONFIG)
  set(cudax_CONFIG "${CMAKE_CURRENT_LIST_FILE}")
endif()
find_package_handle_standard_args(cudax CONFIG_MODE)
