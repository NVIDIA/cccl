#
# find_package(CUB) config file.
#
# Defines a CUB::CUB target that may be linked from user projects to include
# CUB.

if (TARGET CUB::CUB)
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

if (${CMAKE_FIND_PACKAGE_NAME}_FIND_QUIETLY)
  set(
    _CUB_QUIET
    ON
    CACHE INTERNAL
    "Quiet mode enabled for CUB find_package calls."
    FORCE
  )
  set(_CUB_QUIET_FLAG "QUIET" CACHE INTERNAL "" FORCE)
else()
  set(
    _CUB_QUIET
    OFF
    CACHE INTERNAL
    "Quiet mode enabled for CUB find_package calls."
    FORCE
  )
  set(_CUB_QUIET_FLAG "" CACHE INTERNAL "" FORCE)
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
      REQUIRED
      CONFIG
      ${_CUB_QUIET_FLAG}
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
  if (TARGET Thrust::libcudacxx)
    # Prefer the same libcudacxx as Thrust, if available:
    _cub_declare_interface_alias(CUB::libcudacxx _CUB_libcudacxx)
    target_link_libraries(_CUB_libcudacxx INTERFACE Thrust::libcudacxx)
  else()
    if (NOT TARGET libcudacxx::libcudacxx)
      find_package(
        libcudacxx
        ${CUB_VERSION}
        EXACT
        CONFIG
        REQUIRED
        ${_CUB_QUIET_FLAG}
        NO_DEFAULT_PATH # Only check the explicit HINTS below:
        HINTS "${CMAKE_CURRENT_LIST_DIR}/../libcudacxx/"
      )
    endif()
    _cub_declare_interface_alias(CUB::libcudacxx _CUB_libcudacxx)
    target_link_libraries(_CUB_libcudacxx INTERFACE libcudacxx::libcudacxx)
  endif()
endif()

#
# Setup targets
#

_cub_declare_interface_alias(CUB::CUB _CUB_CUB)
target_include_directories(_CUB_CUB INTERFACE "${_CUB_INCLUDE_DIR}")
target_link_libraries(_CUB_CUB INTERFACE CUB::libcudacxx)

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
