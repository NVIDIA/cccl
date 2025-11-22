#
# find_package(libcudacxx) config file.
#
# Defines a libcudacxx::libcudacxx target that may be linked from user projects to include
# libcudacxx.

if (TARGET libcudacxx::libcudacxx)
  return()
endif()

set(quiet_flag)
if (${CMAKE_FIND_PACKAGE_NAME}_FIND_QUIETLY)
  set(quiet_flag "QUIET")
endif()

unset(required_flag)
if (${CMAKE_FIND_PACKAGE_NAME}_FIND_REQUIRED_${component})
  set(required_flag "REQUIRED")
endif()

function(_libcudacxx_declare_interface_alias alias_name ugly_name)
  # 1) Only IMPORTED and ALIAS targets can be placed in a namespace.
  # 2) When an IMPORTED library is linked to another target, its include
  #    directories are treated as SYSTEM includes.
  # 3) nvcc will automatically check the CUDA Toolkit include path *before* the
  #    system includes. This means that the Toolkit libcudacxx will *always* be used
  #    during compilation, and the include paths of an IMPORTED libcudacxx::libcudacxx
  #    target will never have any effect.
  # 4) This behavior can be fixed by setting the property NO_SYSTEM_FROM_IMPORTED
  #    on EVERY target that links to libcudacxx::libcudacxx. This would be a burden and a
  #    footgun for our users. Forgetting this would silently pull in the wrong libcudacxx!
  # 5) A workaround is to make a non-IMPORTED library outside of the namespace,
  #    configure it, and then ALIAS it into the namespace (or ALIAS and then
  #    configure, that seems to work too).
  add_library(${ugly_name} INTERFACE)

  add_library(${alias_name} INTERFACE IMPORTED GLOBAL)
  target_link_libraries(${alias_name} INTERFACE ${ugly_name})
endfunction()

# Create the main libcudacxx target now to avoid circular dependency issues when finding deps.
_libcudacxx_declare_interface_alias(libcudacxx::libcudacxx _libcudacxx_libcudacxx)

#
# Find dependencies
#

if (NOT TARGET libcudacxx::Thrust)
  if (NOT TARGET Thrust::Thrust)
    find_package(
      Thrust
      ${libcudacxx_VERSION}
      CONFIG
      ${quiet_flag}
      ${required}
      NO_DEFAULT_PATH # Only check the explicit HINTS below:
      HINTS "${CMAKE_CURRENT_LIST_DIR}/../thrust/"
    )
  endif()
  _libcudacxx_declare_interface_alias(libcudacxx::Thrust _libcudacxx_Thrust)
  target_link_libraries(libcudacxx::Thrust INTERFACE Thrust::Thrust)
endif()

if (NOT TARGET libcudacxx::CUB)
  if (NOT TARGET CUB::CUB)
    find_package(
      CUB
      ${libcudacxx_VERSION}
      CONFIG
      ${quiet_flag}
      ${required}
      NO_DEFAULT_PATH # Only check the explicit HINTS below:
      HINTS "${CMAKE_CURRENT_LIST_DIR}/../cub/"
    )
  endif()
  _libcudacxx_declare_interface_alias(libcudacxx::CUB _libcudacxx_CUB)
  target_link_libraries(libcudacxx::CUB INTERFACE CUB::CUB)
endif()

#
# Setup targets
#

# Pull in the include dir detected by libcudacxx-config-version.cmake
set(
  _libcudacxx_INCLUDE_DIR
  "${_libcudacxx_VERSION_INCLUDE_DIR}"
  CACHE INTERNAL
  "Location of libcudacxx headers."
)
unset(_libcudacxx_VERSION_INCLUDE_DIR CACHE) # Clear tmp variable from cache

target_link_libraries(
  _libcudacxx_libcudacxx
  INTERFACE #
    libcudacxx::Thrust
    libcudacxx::CUB
)
target_include_directories(
  _libcudacxx_libcudacxx
  INTERFACE "${_libcudacxx_INCLUDE_DIR}"
)
target_compile_definitions(
  _libcudacxx_libcudacxx
  INTERFACE $<$<CONFIG:Debug>:CCCL_ENABLE_ASSERTIONS>
)

#
# Standardize version info
#

set(LIBCUDACXX_VERSION ${${CMAKE_FIND_PACKAGE_NAME}_VERSION} CACHE INTERNAL "")
set(
  LIBCUDACXX_VERSION_MAJOR
  ${${CMAKE_FIND_PACKAGE_NAME}_VERSION_MAJOR}
  CACHE INTERNAL
  ""
)
set(
  LIBCUDACXX_VERSION_MINOR
  ${${CMAKE_FIND_PACKAGE_NAME}_VERSION_MINOR}
  CACHE INTERNAL
  ""
)
set(
  LIBCUDACXX_VERSION_PATCH
  ${${CMAKE_FIND_PACKAGE_NAME}_VERSION_PATCH}
  CACHE INTERNAL
  ""
)
set(
  LIBCUDACXX_VERSION_TWEAK
  ${${CMAKE_FIND_PACKAGE_NAME}_VERSION_TWEAK}
  CACHE INTERNAL
  ""
)
set(
  LIBCUDACXX_VERSION_COUNT
  ${${CMAKE_FIND_PACKAGE_NAME}_VERSION_COUNT}
  CACHE INTERNAL
  ""
)

include(FindPackageHandleStandardArgs)
if (NOT libcudacxx_CONFIG)
  set(libcudacxx_CONFIG "${CMAKE_CURRENT_LIST_FILE}")
endif()
find_package_handle_standard_args(libcudacxx CONFIG_MODE)
