# This file contains checks that ensure a supported build configuration is provided for CCCL.
# These checks are only enforced when building CCCL tests, examples, etc. and are not required
# for users of CCCL.

# The default CXX/CUDA standard to use if none is specified:
set(_cccl_default_dialect 17)

function(cccl_dev_build_checks)
  # Similarly, we expect the CXX and CUDA standards to match, if either is set:
  if (CMAKE_CXX_STANDARD OR CMAKE_CUDA_STANDARD)
    if (NOT CMAKE_CXX_STANDARD EQUAL CMAKE_CUDA_STANDARD)
      message(
        FATAL_ERROR
        "CCCL developer builds require that CMAKE_CXX_STANDARD matches "
        "CMAKE_CUDA_STANDARD when either is set:\n"
        "CMAKE_CXX_STANDARD: ${CMAKE_CXX_STANDARD}\n"
        "CMAKE_CUDA_STANDARD: ${CMAKE_CUDA_STANDARD}\n"
        "Rerun cmake with:\n"
        "\t\"-DCMAKE_CUDA_STANDARD=<std> -DCMAKE_CXX_STANDARD=<std>\"."
      )
    endif()
  else()
    # Neither is set; initialize to a default of 20.
    message(
      VERBOSE
      "Setting CMAKE_CXX_STANDARD and CMAKE_CUDA_STANDARD to CCCL default of ${_cccl_default_dialect}."
    )
    set(CMAKE_CXX_STANDARD ${_cccl_default_dialect})
    set(CMAKE_CUDA_STANDARD ${_cccl_default_dialect})
    set(CMAKE_CXX_STANDARD ${CMAKE_CXX_STANDARD} PARENT_SCOPE)
    set(CMAKE_CUDA_STANDARD ${CMAKE_CUDA_STANDARD} PARENT_SCOPE)
  endif()

  message(STATUS "CMAKE_CXX_STANDARD: ${CMAKE_CXX_STANDARD}")
  message(STATUS "CMAKE_CUDA_STANDARD: ${CMAKE_CUDA_STANDARD}")
endfunction()
