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

foreach(component ${${CMAKE_FIND_PACKAGE_NAME}_FIND_COMPONENTS})
  unset(req)
  if (${CMAKE_FIND_PACKAGE_NAME}_FIND_REQUIRED_${component})
    set(cccl_comp_required_flag "REQUIRED")
  endif()

  if(component STREQUAL "libcudacxx")
    find_package(libcudacxx ${CCCL_VERSION} EXACT CONFIG
      ${cccl_quiet_flag}
      ${cccl_comp_required_flag}
      NO_DEFAULT_PATH # Only check the explicit HINTS below:
      HINTS
        "${cccl_cmake_dir}/../../../libcudacxx/lib/cmake/" # Source layout (GitHub)
        "${cccl_cmake_dir}/.."                             # Install layout
      )
  elseif(component STREQUAL "CUB")
    find_package(CUB ${CCCL_VERSION} EXACT CONFIG
      ${cccl_quiet_flag}
      ${cccl_comp_required_flag}
      NO_DEFAULT_PATH # Only check the explicit HINTS below:
      HINTS
        "${cccl_cmake_dir}/../../../cub/cub/cmake/" # Source layout (GitHub)
        "${cccl_cmake_dir}/.."                      # Install layout
    )
  elseif(component STREQUAL "Thrust")
    find_package(Thrust ${CCCL_VERSION} EXACT CONFIG
      ${cccl_quiet_flag}
      ${cccl_comp_required_flag}
      NO_DEFAULT_PATH # Only check the explicit HINTS below:
      HINTS
        "${cccl_cmake_dir}/../../../thrust/thrust/cmake/" # Source layout (GitHub)
        "${cccl_cmake_dir}/.."                            # Install layout
    )
  else()
    message(FATAL_ERROR "Invalid CCCL component requested: '${component}'")
  endif()
endforeach()

include(FindPackageHandleStandardArgs)
if (NOT CCCL_CONFIG)
  set(CCCL_CONFIG "${CMAKE_CURRENT_LIST_FILE}")
endif()
find_package_handle_standard_args(CCCL CONFIG_MODE)
