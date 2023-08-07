#
# find_package(CCCL) config file.
#
# Imports the Thrust, CUB, and libcudacxx components of the NVIDIA
# CUDA/C++ Core Libraries.

set(cccl_cmake_dir "${CMAKE_CURRENT_LIST_DIR}")

if (${CMAKE_FIND_PACKAGE_NAME}_FIND_QUIETLY)
  set(cccl_quiet_flag "QUIET")
else()
  set(cccl_quiet_flag "")
endif()

if (DEFINED ${CMAKE_FIND_PACKAGE_NAME}_FIND_COMPONENTS AND
    ${CMAKE_FIND_PACKAGE_NAME}_FIND_COMPONENTS)
  set(components ${${CMAKE_FIND_PACKAGE_NAME}_FIND_COMPONENTS})
else()
  set(components Thrust CUB libcudacxx)
endif()

if (NOT TARGET CCCL::CCCL)
  add_library(CCCL::CCCL INTERFACE IMPORTED GLOBAL)
endif()

foreach(component IN LISTS components)
  string(TOLOWER "${component}" component_lower)

  unset(req)
  if (${CMAKE_FIND_PACKAGE_NAME}_FIND_REQUIRED_${component})
    set(cccl_comp_required_flag "REQUIRED")
  endif()

  if(component_lower STREQUAL "libcudacxx")
    find_package(libcudacxx ${CCCL_VERSION} EXACT CONFIG
      ${cccl_quiet_flag}
      ${cccl_comp_required_flag}
      NO_DEFAULT_PATH # Only check the explicit HINTS below:
      HINTS
        "${cccl_cmake_dir}/../../../libcudacxx/lib/cmake/" # Source layout (GitHub)
        "${cccl_cmake_dir}/.."                             # Install layout
    )
    # Can't alias other alias targets, so use the uglified target name instead
    # of libcudacxx::libcudacxx:
    if (TARGET _libcudacxx_libcudacxx AND NOT TARGET CCCL::libcudacxx)
      add_library(CCCL::libcudacxx ALIAS _libcudacxx_libcudacxx)
      target_link_libraries(CCCL::CCCL INTERFACE CCCL::libcudacxx)
    endif()
  elseif(component_lower STREQUAL "cub")
    find_package(CUB ${CCCL_VERSION} EXACT CONFIG
      ${cccl_quiet_flag}
      ${cccl_comp_required_flag}
      NO_DEFAULT_PATH # Only check the explicit HINTS below:
      HINTS
        "${cccl_cmake_dir}/../../../cub/cub/cmake/" # Source layout (GitHub)
        "${cccl_cmake_dir}/.."                      # Install layout
    )
    # Can't alias other alias targets, so use the uglified target name instead
    # of CUB::CUB:
    if (TARGET _CUB_CUB AND NOT TARGET CCCL::CUB)
      add_library(CCCL::CUB ALIAS _CUB_CUB)
      target_link_libraries(CCCL::CCCL INTERFACE CCCL::CUB)
    endif()
  elseif(component_lower STREQUAL "thrust")
    find_package(Thrust ${CCCL_VERSION} EXACT CONFIG
      ${cccl_quiet_flag}
      ${cccl_comp_required_flag}
      NO_DEFAULT_PATH # Only check the explicit HINTS below:
      HINTS
        "${cccl_cmake_dir}/../../../thrust/thrust/cmake/" # Source layout (GitHub)
        "${cccl_cmake_dir}/.."                            # Install layout
    )

    if (TARGET Thrust::Thrust AND NOT CCCL::Thrust)
      # By default, configure a CCCL::Thrust target with host=cpp device=cuda
      option(CCCL_ENABLE_DEFAULT_THRUST_TARGET
        "Create a CCCL::Thrust target using CCCL_THRUST_[HOST|DEVICE]_SYSTEM."
        ON
      )
      mark_as_advanced(CCCL_ENABLE_DEFAULT_THRUST_TARGET)
      if (CCCL_ENABLE_DEFAULT_THRUST_TARGET)
        thrust_create_target(CCCL::Thrust FROM_OPTIONS
          HOST_OPTION CCCL_THRUST_HOST_SYSTEM
          DEVICE_OPTION CCCL_THRUST_DEVICE_SYSTEM
          HOST_OPTION_DOC
            "Host system for CCCL::Thrust target."
          DEVICE_OPTION_DOC
            "Device system for CCCL::Thrust target."
          ADVANCED
        )
        target_link_libraries(CCCL::CCCL INTERFACE CCCL::Thrust)
      endif()
    endif()
  else()
    message(FATAL_ERROR "Invalid CCCL component requested: '${component}'")
  endif()
endforeach()

include(FindPackageHandleStandardArgs)
if (NOT CCCL_CONFIG)
  set(CCCL_CONFIG "${CMAKE_CURRENT_LIST_FILE}")
endif()
find_package_handle_standard_args(CCCL CONFIG_MODE)
