if (TARGET CudaNext::CudaNext)
  return()
endif()

#
# Setup dependencies
#

set(cn_quiet_flag "")
if (${CMAKE_FIND_PACKAGE_NAME}_FIND_QUIETLY)
  set(cn_quiet_flag "QUIET")
endif()

set(cn_cmake_dir "${CMAKE_CURRENT_LIST_DIR}")
set(cn_prefix_dir "${cn_cmake_dir}/../../..")
set(cn_include_dir "${cn_prefix_dir}/include")

if (NOT TARGET CudaNext::libcudacxx)
  if (NOT TARGET libcudacxx::libcudacxx)
    # First do a non-required search for any co-packaged versions.
    # These are preferred.
    find_package(libcudacxx ${CudaNext_VERSION} CONFIG
      ${cn_quiet_flag}
      NO_DEFAULT_PATH # Only check the explicit HINTS below:
      HINTS
        "${cn_prefix_dir}/../libcudacxx" # Source layout
        "${cn_cmake_dir}/.."             # Install layout
    )

    # A second required search allows externally packaged to be used and fails if
    # no suitable package exists.
    find_package(libcudacxx ${CudaNext_VERSION} CONFIG
      REQUIRED
      ${cn_quiet_flag}
    )
  endif()
  add_library(CudaNext::libcudacxx ALIAS libcudacxx::libcudacxx)
endif()

# Allow non-imported targets to be used. This treats CudaNext headers as non-system includes,
# exposing warnings in them. Used when building CudaNext tests.
if (CudaNext_NO_IMPORTED_TARGETS)
  # INTERFACE libraries cannot be namespaced; ALIAS libraries can.
  add_library(_CudaNext_CudaNext INTERFACE)
  add_library(CudaNext::CudaNext ALIAS _CudaNext_CudaNext)
  set(cn_target_name _CudaNext_CudaNext)
  else()
  add_library(CudaNext::CudaNext INTERFACE IMPORTED GLOBAL)
  set(cn_target_name CudaNext::CudaNext)
endif()

target_compile_features(${cn_target_name} INTERFACE cxx_std_17)
target_include_directories(${cn_target_name} INTERFACE "${cn_include_dir}")
target_link_libraries(${cn_target_name} INTERFACE CudaNext::libcudacxx)

# Expose version info globally through cache variables:
set(CudaNext_VERSION ${${CMAKE_FIND_PACKAGE_NAME}_VERSION} CACHE INTERNAL "" FORCE)
set(CudaNext_VERSION_MAJOR ${${CMAKE_FIND_PACKAGE_NAME}_VERSION_MAJOR} CACHE INTERNAL "" FORCE)
set(CudaNext_VERSION_MINOR ${${CMAKE_FIND_PACKAGE_NAME}_VERSION_MINOR} CACHE INTERNAL "" FORCE)
set(CudaNext_VERSION_PATCH ${${CMAKE_FIND_PACKAGE_NAME}_VERSION_PATCH} CACHE INTERNAL "" FORCE)
set(CudaNext_VERSION_TWEAK ${${CMAKE_FIND_PACKAGE_NAME}_VERSION_TWEAK} CACHE INTERNAL "" FORCE)
set(CudaNext_VERSION_COUNT ${${CMAKE_FIND_PACKAGE_NAME}_VERSION_COUNT} CACHE INTERNAL "" FORCE)

include(FindPackageHandleStandardArgs)
if (NOT CudaNext_CONFIG)
  set(CudaNext_CONFIG "${CMAKE_CURRENT_LIST_FILE}")
endif()
find_package_handle_standard_args(CudaNext CONFIG_MODE)
