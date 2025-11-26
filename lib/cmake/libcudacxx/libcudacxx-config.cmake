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
  add_library(${alias_name} ALIAS ${ugly_name})
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
  target_link_libraries(_libcudacxx_Thrust INTERFACE Thrust::Thrust)
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
  target_link_libraries(_libcudacxx_CUB INTERFACE CUB::CUB)
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

# Detect whether the host compiler is MSVC
set(detected_msvc_host FALSE)
set(detected_msvc_host_version)

if (NOT CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
  # Only nvcc uses a host compiler.
  set(detected_msvc_host FALSE)
elseif (
  DEFINED CCCL_OVERRIDE_MSVC_HOST_CHECK
  AND DEFINED CCCL_OVERRIDE_MSVC_HOST_VERSION
)
  # Set CCCL_OVERRIDE_MSVC_HOST_CHECK to TRUE or FALSE to force the value of
  # detected_msvc_host. We provide this because host compiler id detection is not
  # robustly available in all CMake versions and generator combinations and users
  # may need an escape hatch to avoid adding incorrect flags.
  # CCCL_OVERRIDE_MSVC_HOST_VERSION should also be set to
  # the version number of the MSVC host compiler (e.g., 19.29).
  set(detected_msvc_host ${CCCL_OVERRIDE_MSVC_HOST_CHECK})
  set(detected_msvc_host_version ${CCCL_OVERRIDE_MSVC_HOST_VERSION})
else()
  # gersemi: off
  if ((CMAKE_VERSION VERSION_GREATER_EQUAL 3.31) AND
      (NOT CMAKE_GENERATOR MATCHES "Visual Studio"))
    # gersemi: on
    # If CMake >= 3.31 and the CMake generator is not Visual Studio,
    # CMAKE_CUDA_HOST_COMPILER_ID is available:
    if (CMAKE_CUDA_HOST_COMPILER_ID STREQUAL "MSVC")
      set(detected_msvc_host TRUE)
      set(detected_msvc_host_version ${CMAKE_CUDA_HOST_COMPILER_VERSION})
    endif()
  else()
    # For CMake < 3.31 or Visual Studio generators, fall back to checking CMAKE_CXX_COMPILER_ID.
    # Usually windows nvcc builds use MSVC so this should normally work.
    # Use CCCL_OVERRIDE_MSVC_HOST_CHECK to override for weird edge cases.
    if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
      set(detected_msvc_host TRUE)
      set(detected_msvc_host_version ${CMAKE_CXX_COMPILER_VERSION})
    endif()
  endif()
endif()

if (detected_msvc_host)
  # We require the conforming __cplusplus behavior:
  target_compile_options(
    _libcudacxx_libcudacxx
    INTERFACE
      "$<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-Xcompiler=/Zc:__cplusplus>"
      "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/Zc:__cplusplus>"
  )

  # libcudacxx requires the conforming preprocessor on MSVC builds.
  if (detected_msvc_host_version GREATER_EQUAL 19.25)
    target_compile_options(
      _libcudacxx_libcudacxx
      INTERFACE
        "$<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-Xcompiler=/Zc:preprocessor>"
        "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/Zc:preprocessor>"
    )
  elseif (detected_msvc_host_version GREATER_EQUAL 19.15)
    # Older version of this flag:
    target_compile_options(
      _libcudacxx_libcudacxx
      INTERFACE
        "$<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-Xcompiler=/experimental:preprocessor>"
        "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/experimental:preprocessor>"
    )
  else()
    # Shouldn't really happen, current CCCL requires at least MSVC 19.20+ (as of CCCL 3.x).
    # For completeness:
    message(
      WARNING
      "Detected MSVC host compiler unsupported by CCCL: ${detected_msvc_host_version}."
    )
  endif()
endif()

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
