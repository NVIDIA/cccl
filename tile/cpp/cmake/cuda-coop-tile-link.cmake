# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# TileLib is consumed by CUDA compilation, so its replacement must rebuild objects.
function(cuda_coop_tile_link target)
  if (NOT TARGET "${target}")
    message(FATAL_ERROR "cuda_coop_tile_link: target does not exist: ${target}")
  endif()
  get_target_property(_sources "${target}" SOURCES)
  set(_cuda_sources)
  foreach (_source IN LISTS _sources)
    get_source_file_property(
      _language
      "${_source}"
      TARGET_DIRECTORY "${target}"
      LANGUAGE
    )
    if (_language STREQUAL "CUDA")
      list(APPEND _cuda_sources "${_source}")
    endif()
  endforeach()
  if (NOT _cuda_sources)
    message(
      FATAL_ERROR
      "cuda_coop_tile_link: ${target} has no CUDA sources; call after adding its sources"
    )
  endif()
  get_target_property(_tilelib cuda-coop-tile::tile CUDA_COOP_TILE_LIBRARY)
  target_link_libraries("${target}" PRIVATE cuda-coop-tile::tile)
  set_property(
    SOURCE ${_cuda_sources} TARGET_DIRECTORY "${target}"
    APPEND
    PROPERTY OBJECT_DEPENDS "${_tilelib}"
  )
endfunction()
