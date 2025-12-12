#
# find_package(libcudacxx) config file.
#
# Defines a libcudacxx::libcudacxx target that may be linked from user projects to include
# libcudacxx.

if (TARGET libcudacxx::libcudacxx)
  # This isn't the first time we've been included -- double check the enabled languages
  # and re-run compiler checks for any new ones.
  libcudacxx_update_language_compat_flags()

  include(FindPackageHandleStandardArgs)
  if (NOT libcudacxx_CONFIG)
    set(libcudacxx_CONFIG "${CMAKE_CURRENT_LIST_FILE}")
  endif()
  find_package_handle_standard_args(libcudacxx CONFIG_MODE)
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

function(_libcudacxx_define_internal_global_property prop_name)
  # Need to define docs options for CMake < 3.23 (optional in later versions)
  define_property(
    GLOBAL
    PROPERTY ${prop_name}
    BRIEF_DOCS "Internal libcudacxx property: ${prop_name}."
    FULL_DOCS "Internal libcudacxx property: ${prop_name}."
  )
endfunction()

# We cannot test for MSVC version if the CXX or CUDA languages aren't enabled, because
# the CMAKE_[CXX|CUDA_HOST]_COMPILER_[ID|VERSION] variables won't exist.
# Just call find_package(libcudacxx) again after enabling languages to rediscover.
function(libcudacxx_update_language_compat_flags)
  # Track which languages have already been checked:
  # gersemi: off
  get_property(cxx_checked GLOBAL PROPERTY _libcudacxx_cxx_checked DEFINED)
  get_property(cuda_checked GLOBAL PROPERTY _libcudacxx_cuda_checked DEFINED)
  get_property(cxx_warned GLOBAL PROPERTY _libcudacxx_cxx_warned DEFINED)
  get_property(cuda_warned GLOBAL PROPERTY _libcudacxx_cuda_warned DEFINED)
  get_property(mismatch_warned GLOBAL PROPERTY _libcudacxx_mismatch_warned DEFINED)
  get_property(langs GLOBAL PROPERTY ENABLED_LANGUAGES)

  message(DEBUG "libcudacxx: Languages: ${langs}")
  message(DEBUG "libcudacxx:   cxx_checked: ${cxx_checked}")
  message(DEBUG "libcudacxx:   cuda_checked: ${cuda_checked}")
  message(DEBUG "libcudacxx:   cxx_warned: ${cxx_warned}")
  message(DEBUG "libcudacxx:   cuda_warned: ${cuda_warned}")
  message(DEBUG "libcudacxx:   CMAKE_VERSION: ${CMAKE_VERSION}")
  message(DEBUG "libcudacxx:   CMAKE_GENERATOR: ${CMAKE_GENERATOR}")
  message(DEBUG "libcudacxx:   CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")
  message(DEBUG "libcudacxx:   CMAKE_CXX_COMPILER_ID: ${CMAKE_CXX_COMPILER_ID}")
  message(DEBUG "libcudacxx:   CMAKE_CXX_COMPILER_VERSION: ${CMAKE_CXX_COMPILER_VERSION}")
  message(DEBUG "libcudacxx:   CMAKE_CUDA_HOST_COMPILER: ${CMAKE_CUDA_HOST_COMPILER}")
  message(DEBUG "libcudacxx:   CMAKE_CUDA_HOST_COMPILER_ID: ${CMAKE_CUDA_HOST_COMPILER_ID}")
  message(DEBUG "libcudacxx:   CMAKE_CUDA_HOST_COMPILER_VERSION: ${CMAKE_CUDA_HOST_COMPILER_VERSION}")
  # gersemi: on

  if (NOT cxx_warned AND NOT CXX IN_LIST langs)
    # gersemi: off
    message(VERBOSE "libcudacxx: - CXX language not enabled.")
    message(VERBOSE "libcudacxx:   /Zc:__cplusplus and /Zc:preprocessor flags may not be automatically added to CXX targets.")
    message(VERBOSE "libcudacxx:   Call find_package(CCCL) again after enabling CXX to enable compatibility flags.")
    # gersemi: on
    _libcudacxx_define_internal_global_property(_libcudacxx_cxx_warned)
  endif()

  if (NOT cuda_warned AND NOT CUDA IN_LIST langs)
    # gersemi: off
    message(VERBOSE "libcudacxx: - CUDA language not enabled.")
    message(VERBOSE "libcudacxx:   /Zc:__cplusplus and /Zc:preprocessor flags may not be automatically added to CUDA targets.")
    message(VERBOSE "libcudacxx:   Call find_package(CCCL) again after enabling CUDA to enable compatibility flags.")
    # gersemi: on
    _libcudacxx_define_internal_global_property(_libcudacxx_cuda_warned)
  endif()

  if (CXX IN_LIST langs)
    set(msvc_cxx_id ${CMAKE_CXX_COMPILER_ID})
    set(msvc_cxx_version ${CMAKE_CXX_COMPILER_VERSION})
  endif()

  if (CUDA IN_LIST langs)
    option(
      libcudacxx_MISMATCHED_HOST_COMPILER
      "Set to true if CXX / CUDA_HOST compilers are different."
      FALSE
    )
    mark_as_advanced(libcudacxx_MISMATCHED_HOST_COMPILER)
    if (
      (NOT CMAKE_GENERATOR MATCHES "Visual Studio")
      AND (CMAKE_VERSION VERSION_GREATER_EQUAL 3.31)
    )
      # These aren't defined with VS gens or older cmake versions:
      set(msvc_cuda_host_id ${CMAKE_CUDA_HOST_COMPILER_ID})
      set(msvc_cuda_host_version ${CMAKE_CUDA_HOST_COMPILER_VERSION})
    elseif (CMAKE_CUDA_HOST_COMPILER STREQUAL CMAKE_CXX_COMPILER)
      # Same path, same compiler:
      set(msvc_cuda_host_id ${CMAKE_CXX_COMPILER_ID})
      set(msvc_cuda_host_version ${CMAKE_CXX_COMPILER_VERSION})
      # gersemi: off
    elseif ((NOT mismatch_warned) AND
            (NOT CMAKE_CUDA_HOST_COMPILER) AND
            (NOT libcudacxx_MISMATCHED_HOST_COMPILER))
      # For CMake < 3.31, we cannot reliably detect the CUDA host compiler ID.
      # Assume that the CUDA host compiler is the same as the CXX compiler.
      # Usually a safe assumption but provide an escape hatch for edge cases.
      message(STATUS "libcudacxx: - Assuming CUDA host compiler is the same as CXX compiler.")
      message(STATUS "libcudacxx:   Set libcudacxx_MISMATCHED_HOST_COMPILER=TRUE to disable this.")
      _libcudacxx_define_internal_global_property(_libcudacxx_mismatch_warned)
      set(msvc_cuda_host_id ${CMAKE_CXX_COMPILER_ID})
      set(msvc_cuda_host_version ${CMAKE_CXX_COMPILER_VERSION})
    endif()
    # gersemi: on
  endif()

  function(_libcudacxx_get_msvc_flags_for_version out_var msvc_version)
    set(flags "/Zc:__cplusplus")
    if (msvc_version GREATER_EQUAL 19.25)
      list(APPEND flags "/Zc:preprocessor")
    elseif (msvc_version GREATER_EQUAL 19.15)
      list(APPEND flags "/experimental:preprocessor")
    endif()
    set(${out_var} "${flags}" PARENT_SCOPE)
  endfunction()

  if (NOT cxx_checked AND DEFINED msvc_cxx_id)
    if (msvc_cxx_id STREQUAL "MSVC")
      _libcudacxx_get_msvc_flags_for_version(cxx_flags "${msvc_cxx_version}")
      foreach (flag IN LISTS cxx_flags)
        target_compile_options(
          _libcudacxx_libcudacxx
          INTERFACE "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:${flag}>"
        )
        message(
          STATUS
          "libcudacxx: - Added CXX compile option for MSVC: ${flag}"
        )
      endforeach()
    endif()
    _libcudacxx_define_internal_global_property(_libcudacxx_cxx_checked)
  endif()

  if (NOT cuda_checked AND DEFINED msvc_cuda_host_id)
    if (msvc_cuda_host_id STREQUAL "MSVC")
      _libcudacxx_get_msvc_flags_for_version(cuda_flags "${msvc_cuda_host_version}")
      foreach (flag IN LISTS cuda_flags)
        target_compile_options(
          _libcudacxx_libcudacxx
          INTERFACE "$<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-Xcompiler=${flag}>"
        )
        message(
          STATUS
          "libcudacxx: - Added CUDA host compile option for MSVC: ${flag}"
        )
      endforeach()
    endif()
    _libcudacxx_define_internal_global_property(_libcudacxx_cuda_checked)
  endif()
endfunction()

libcudacxx_update_language_compat_flags()

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
