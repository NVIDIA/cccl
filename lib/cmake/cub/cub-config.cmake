#
# find_package(CUB) config file.
#
# Defines a CUB::CUB target that may be linked from user projects to include
# CUB.

if (TARGET CUB::CUB)
  # In case new languages have been enabled:
  libcudacxx_update_language_compat_flags()

  include(FindPackageHandleStandardArgs)
  if (NOT CUB_CONFIG)
    set(CUB_CONFIG "${CMAKE_CURRENT_LIST_FILE}")
  endif()
  find_package_handle_standard_args(CUB CONFIG_MODE)
  return()
endif()

function(_cub_declare_interface_alias alias_name ugly_name)
  # 1) Only IMPORTED and ALIAS targets can be placed in a namespace.
  # 2) When an IMPORTED library is linked to another target, its include
  #    directories are treated as SYSTEM includes.
  # 3) nvcc will automatically check the CUDA Toolkit include path *before* the
  #    system includes. This means that the Toolkit CUB will *always* be used
  #    during compilation, and the include paths of an IMPORTED CUB::CUB
  #    target will never have any effect.
  # 4) This behavior can be fixed by setting the property NO_SYSTEM_FROM_IMPORTED
  #    on EVERY target that links to CUB::CUB. This would be a burden and a
  #    footgun for our users. Forgetting this would silently pull in the wrong CUB!
  # 5) A workaround is to make a non-IMPORTED library outside of the namespace,
  #    configure it, and then ALIAS it into the namespace (or ALIAS and then
  #    configure, that seems to work too).
  add_library(${ugly_name} INTERFACE)
  add_library(${alias_name} ALIAS ${ugly_name})
endfunction()

# Create the main cub target now to avoid circular dependency issues when finding deps.
_cub_declare_interface_alias(CUB::CUB _CUB_CUB)

#
# Setup some internal cache variables
#

# Pull in the include dir detected by cub-config-version.cmake
set(
  _CUB_INCLUDE_DIR
  "${_CUB_VERSION_INCLUDE_DIR}"
  CACHE INTERNAL
  "Location of CUB headers."
  FORCE
)
unset(_CUB_VERSION_INCLUDE_DIR CACHE) # Clear tmp variable from cache

set(quiet_flag)
if (${CMAKE_FIND_PACKAGE_NAME}_FIND_QUIETLY)
  set(quiet_flag "QUIET")
endif()

unset(required_flag)
if (${CMAKE_FIND_PACKAGE_NAME}_FIND_REQUIRED_${component})
  set(required_flag "REQUIRED")
endif()

#
# Setup dependencies
#

if (NOT TARGET CUB::Thrust)
  if (NOT TARGET Thrust::Thrust)
    find_package(
      Thrust
      ${CUB_VERSION}
      EXACT
      CONFIG
      ${required_flag}
      ${quiet_flag}
      NO_DEFAULT_PATH
      HINTS "${CMAKE_CURRENT_LIST_DIR}/../thrust/"
    )
  endif()
  _cub_declare_interface_alias(CUB::Thrust _CUB_Thrust)
  # Just link Thrust::Thrust -- this is the minimal target that only provides
  # headers with no host/device system setup, which is all CUB needs.
  target_link_libraries(_CUB_Thrust INTERFACE Thrust::Thrust)
endif()

if (NOT TARGET CUB::libcudacxx)
  if (NOT TARGET libcudacxx::libcudacxx)
    find_package(
      libcudacxx
      ${CUB_VERSION}
      EXACT
      CONFIG
      ${required_flag}
      ${quiet_flag}
      NO_DEFAULT_PATH # Only check the explicit HINTS below:
      HINTS "${CMAKE_CURRENT_LIST_DIR}/../libcudacxx/"
    )
  endif()
  _cub_declare_interface_alias(CUB::libcudacxx _CUB_libcudacxx)
  target_link_libraries(_CUB_libcudacxx INTERFACE libcudacxx::libcudacxx)
endif()

#
# Setup targets
#

target_include_directories(_CUB_CUB INTERFACE "${_CUB_INCLUDE_DIR}")
target_link_libraries(
  _CUB_CUB
  INTERFACE #
    CUB::libcudacxx
    CUB::Thrust
)

function(_cub_test_flag_option flag)
  if (CCCL_${flag} OR CUB_${flag} OR THRUST_${flag})
    target_compile_definitions(_CUB_CUB INTERFACE "CCCL_${flag}")
  endif()
endfunction()
_cub_test_flag_option(IGNORE_DEPRECATED_API)
_cub_test_flag_option(IGNORE_DEPRECATED_CPP_DIALECT)
_cub_test_flag_option(IGNORE_DEPRECATED_CPP_11)
_cub_test_flag_option(IGNORE_DEPRECATED_CPP_14)
_cub_test_flag_option(IGNORE_DEPRECATED_COMPILER)

#
# Standardize version info
#

set(CUB_VERSION ${${CMAKE_FIND_PACKAGE_NAME}_VERSION} CACHE INTERNAL "" FORCE)
set(
  CUB_VERSION_MAJOR
  ${${CMAKE_FIND_PACKAGE_NAME}_VERSION_MAJOR}
  CACHE INTERNAL
  ""
  FORCE
)
set(
  CUB_VERSION_MINOR
  ${${CMAKE_FIND_PACKAGE_NAME}_VERSION_MINOR}
  CACHE INTERNAL
  ""
  FORCE
)
set(
  CUB_VERSION_PATCH
  ${${CMAKE_FIND_PACKAGE_NAME}_VERSION_PATCH}
  CACHE INTERNAL
  ""
  FORCE
)
set(
  CUB_VERSION_TWEAK
  ${${CMAKE_FIND_PACKAGE_NAME}_VERSION_TWEAK}
  CACHE INTERNAL
  ""
  FORCE
)
set(
  CUB_VERSION_COUNT
  ${${CMAKE_FIND_PACKAGE_NAME}_VERSION_COUNT}
  CACHE INTERNAL
  ""
  FORCE
)

include(FindPackageHandleStandardArgs)
if (NOT CUB_CONFIG)
  set(CUB_CONFIG "${CMAKE_CURRENT_LIST_FILE}")
endif()
find_package_handle_standard_args(CUB CONFIG_MODE)
