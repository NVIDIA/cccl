# This file provides utilities to handle special CMAKE_CUDA_ARCHITECTURES lists for CCCL.
#
# If CMAKE_CUDA_ARCHITECTURES is set to one of the following values, it will be replaced
# as described:
#
# 'all-cccl': All architectures known to the current NVCC above minimum_cccl_arch.
#
# 'all-major-cccl': All major architectures known to the current NVCC above minimum_cccl_arch,
# plus 'minimum_cccl_arch'.
#
# For example on 12.9:
#   all: 50-real;52-real;53-real;60-real;61-real;62-real;70-real;72-real;75-real;80-real;86-real;87-real;89-real;90-real;100-real;101-real;103-real;120-real;121-real;121-virtual
#   all-cccl: 75-real;80-real;86-real;87-real;89-real;90-real;100-real;101-real;103-real;120-real;121-real;121-virtual
#   all-major: 50-real;60-real;70-real;80-real;90-real;100-real;120-real;120-virtual
#   all-major-cccl: 75-real;80-real;90-real;100-real;120-real;120-virtual

# We don't support arches below what the latest CTK release supports:
set(minimum_cccl_arch 75) # 13.x dropped below Turing

# Check CMAKE_CUDA_ARCHITECTURES for special CCCL values and update as described above.
function(cccl_check_cuda_architectures)
  if (CMAKE_CUDA_ARCHITECTURES MATCHES "-cccl$")
    message(
      STATUS
      "Detected special CCCL arch request: CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}"
    )

    _cccl_detect_nvcc_arch_support(arches)
    _cccl_filter_to_supported_arches(arches)

    if (CMAKE_CUDA_ARCHITECTURES STREQUAL "all-major-cccl")
      _cccl_filter_to_all_major_cccl(arches)
    elseif (CMAKE_CUDA_ARCHITECTURES STREQUAL "all-cccl")
      # No further filtering needed, just use the arches as is.
    else()
      message(
        FATAL_ERROR
        "Invalid CMAKE_CUDA_ARCHITECTURES value: ${CMAKE_CUDA_ARCHITECTURES}"
      )
    endif()

    _cccl_add_real_virtual_arch_tags(arches)
    message(STATUS "Replacing with CMAKE_CUDA_ARCHITECTURES=${arches}")
    set(
      CMAKE_CUDA_ARCHITECTURES
      "${arches}"
      CACHE STRING
      "CUDA architectures for CCCL"
      FORCE
    )
  endif()
endfunction()

# Query nvcc --help to determine which architectures are supported.
function(_cccl_detect_nvcc_arch_support arches_var)
  find_package(CUDAToolkit)
  if (NOT CUDAToolkit_FOUND)
    message(
      FATAL_ERROR
      "CUDAToolkit not found, '${CMAKE_CUDA_ARCHITECTURES}' arch detection failed."
    )
  endif()

  execute_process(
    COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}" --help
    OUTPUT_VARIABLE nvcc_help_output
    COMMAND_ERROR_IS_FATAL ANY
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  string(REGEX MATCHALL "compute_[0-9]+" supported_arches "${nvcc_help_output}")
  string(REPLACE "compute_" "" supported_arches "${supported_arches}")
  list(SORT supported_arches COMPARE NATURAL)
  list(REMOVE_DUPLICATES supported_arches)
  message(VERBOSE "NVCC supports: ${supported_arches}")
  set(${arches_var} ${supported_arches} PARENT_SCOPE)
endfunction()

# Remove all arches < minimum_cccl_arch
function(_cccl_filter_to_supported_arches arches_var)
  set(cccl_arches "")
  foreach (arch IN LISTS ${arches_var})
    if (arch GREATER_EQUAL minimum_cccl_arch)
      list(APPEND cccl_arches ${arch})
    endif()
  endforeach()
  message(VERBOSE "CCCL supported arches: ${cccl_arches}")
  set(${arches_var} ${cccl_arches} PARENT_SCOPE)
endfunction()

# Convert all-cccl to all-major-cccl.
function(_cccl_filter_to_all_major_cccl arches_var)
  set(major_arches "")
  foreach (arch IN LISTS ${arches_var})
    math(EXPR major "(${arch} / 10) * 10")
    if (major LESS minimum_cccl_arch)
      set(major "${minimum_cccl_arch}")
    endif()
    if (NOT major IN_LIST major_arches)
      list(APPEND major_arches ${major})
    endif()
  endforeach()
  message(VERBOSE "CCCL all-major arches: ${major_arches}")
  set(${arches_var} ${major_arches} PARENT_SCOPE)
endfunction()

function(_cccl_add_real_virtual_arch_tags arches_var)
  set(tagged_arches "")

  list(POP_BACK ${arches_var} last_arch)

  foreach (arch IN LISTS ${arches_var})
    list(APPEND tagged_arches "${arch}-real")
  endforeach()

  list(APPEND tagged_arches "${last_arch}-real")
  list(APPEND tagged_arches "${last_arch}-virtual")

  message(VERBOSE "CCCL tagged arches: ${tagged_arches}")
  set(${arches_var} ${tagged_arches} PARENT_SCOPE)
endfunction()
