# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(_tilelib_dir "${CMAKE_CURRENT_SOURCE_DIR}/tilelib")
set(_sdk_include "${CUDA_COOP_TILE_SDK_ROOT}/include")
file(
  GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/cuda-coop-tile-cccl-root.txt"
  CONTENT "${CCCL_SOURCE_DIR}\n"
)

add_library(
  cuda_coop_tile_bytecode
  OBJECT
  "${CUDA_COOP_TILE_SDK_ROOT}/src/TileIRBytecodeBuilder.c"
)
set_target_properties(
  cuda_coop_tile_bytecode
  PROPERTIES
    C_STANDARD 11
    C_STANDARD_REQUIRED YES
    POSITION_INDEPENDENT_CODE YES
    C_VISIBILITY_PRESET hidden
)
target_include_directories(
  cuda_coop_tile_bytecode
  PRIVATE "${_sdk_include}" "${_sdk_include}/tile_ir/linkage_provider"
)

add_library(
  cuda_coop_tile_provider
  SHARED
  tilelib/coop_linkage_provider.cpp
  tilelib/provider/metadata_parser.cpp
  tilelib/provider/primitive_spec.cpp
  tilelib/provider/provider_request.cpp
  tilelib/provider/registry.cpp
  tilelib/provider/source_common.cpp
  tilelib/provider/spec_identity.cpp
  tilelib/provider/families/adjacent_difference.cpp
  tilelib/provider/families/discontinuity.cpp
  tilelib/provider/families/emitter.cpp
  tilelib/provider/families/exchange.cpp
  tilelib/provider/families/histogram_rld.cpp
  tilelib/provider/families/merge_sort.cpp
  tilelib/provider/families/radix_rank.cpp
  tilelib/provider/families/radix_sort.cpp
  tilelib/provider/families/reduce.cpp
  tilelib/provider/families/scan.cpp
  tilelib/provider/families/shuffle.cpp
  tilelib/provider/runtime/bundle_builder.cpp
  tilelib/provider/runtime/cache_keys.cpp
  tilelib/provider/runtime/include_paths.cpp
  tilelib/provider/runtime/nvrtc_compiler.cpp
  tilelib/provider/runtime/resource_probe.cpp
  tilelib/provider/runtime/tracing.cpp
  $<TARGET_OBJECTS:cuda_coop_tile_bytecode>
)
target_compile_features(cuda_coop_tile_provider PRIVATE cxx_std_17)
target_include_directories(
  cuda_coop_tile_provider
  PRIVATE "${_tilelib_dir}" "${_sdk_include}"
)
get_filename_component(
  _installed_library_dir
  "${CUDA_COOP_TILE_INSTALL_LIBRARY_DIR}"
  ABSOLUTE
  BASE_DIR "${CMAKE_INSTALL_PREFIX}"
)
get_filename_component(
  _installed_include_dir
  "${CUDA_COOP_TILE_INSTALL_INCLUDE_DIR}/cccl"
  ABSOLUTE
  BASE_DIR "${CMAKE_INSTALL_PREFIX}"
)
file(
  RELATIVE_PATH
  _installed_include_relative
  "${_installed_library_dir}"
  "${_installed_include_dir}"
)
target_compile_definitions(
  cuda_coop_tile_provider
  PRIVATE
    "CUDA_COOP_TILE_INSTALLED_INCLUDE_RELATIVE=\"${_installed_include_relative}\""
)
target_link_libraries(
  cuda_coop_tile_provider
  PRIVATE CUDA::nvrtc ${CMAKE_DL_LIBS}
)
set_target_properties(
  cuda_coop_tile_provider
  PROPERTIES
    OUTPUT_NAME coop_linkage_provider
    BUILD_WITH_INSTALL_RPATH YES
    INSTALL_RPATH ""
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
)

set(_interface "${CMAKE_CURRENT_BINARY_DIR}/cuda_coop_tile.tilebc")
add_custom_command(
  OUTPUT "${_interface}"
  COMMAND
    "${CUDA_COOP_TILE_INTERFACE_TRANSLATOR}" --mlir-to-cudatilebc
    --no-implicit-module --preview=simt --bytecode-version=13.5
    "${_tilelib_dir}/interface/cuda_coop_tile.mlir" -o "${_interface}"
  DEPENDS
    "${_tilelib_dir}/interface/cuda_coop_tile.mlir"
    "${CUDA_COOP_TILE_INTERFACE_TRANSLATOR}"
  VERBATIM
)
add_custom_target(cuda_coop_tile_interface DEPENDS "${_interface}")
set(_tilelib "${CMAKE_CURRENT_BINARY_DIR}/cuda_coop_tile.tilelib")
add_custom_command(
  OUTPUT "${_tilelib}"
  COMMAND
    "${CUDA_COOP_TILE_TILEIRAS}" --preview-create-tile-lib
    "--preview-interface=${_interface}"
    "--preview-implementation=$<TARGET_FILE:cuda_coop_tile_provider>" -o
    "${_tilelib}"
  DEPENDS cuda_coop_tile_provider "${_interface}" "${CUDA_COOP_TILE_TILEIRAS}"
  VERBATIM
)
add_custom_target(cuda_coop_tile_library ALL DEPENDS "${_tilelib}")

add_library(cuda_coop_tile INTERFACE)
add_library(cuda-coop-tile::tile ALIAS cuda_coop_tile)
add_dependencies(cuda_coop_tile cuda_coop_tile_library)
set_property(
  TARGET cuda_coop_tile
  PROPERTY CUDA_COOP_TILE_LIBRARY "${_tilelib}"
)
target_include_directories(
  cuda_coop_tile
  INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CCCL_SOURCE_DIR}/libcudacxx/include"
    "${CCCL_SOURCE_DIR}/cub"
    "${CCCL_SOURCE_DIR}/thrust"
)
target_compile_features(cuda_coop_tile INTERFACE cuda_std_20)
target_compile_options(
  cuda_coop_tile
  INTERFACE
    "$<$<COMPILE_LANGUAGE:CUDA>:--enable-tile>"
    "$<$<COMPILE_LANGUAGE:CUDA>:--tile-preview=simt>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-Xtileiras=-l=\"${_tilelib}\">"
)
if (WIN32)
  target_compile_options(
    cuda_coop_tile
    INTERFACE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>"
  )
endif()
include(cmake/cuda-coop-tile-link.cmake)

install(FILES "${_tilelib}" DESTINATION "${CUDA_COOP_TILE_INSTALL_LIBRARY_DIR}")
install(
  TARGETS cuda_coop_tile_provider
  LIBRARY DESTINATION "${CUDA_COOP_TILE_INSTALL_LIBRARY_DIR}"
  RUNTIME DESTINATION "${CUDA_COOP_TILE_INSTALL_LIBRARY_DIR}"
)
install(DIRECTORY include/ DESTINATION "${CUDA_COOP_TILE_INSTALL_INCLUDE_DIR}")
install(
  DIRECTORY "${CCCL_SOURCE_DIR}/libcudacxx/include/"
  DESTINATION "${CUDA_COOP_TILE_INSTALL_INCLUDE_DIR}/cccl"
)
install(
  DIRECTORY "${CCCL_SOURCE_DIR}/cub/cub" "${CCCL_SOURCE_DIR}/thrust/thrust"
  DESTINATION "${CUDA_COOP_TILE_INSTALL_INCLUDE_DIR}/cccl"
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.cuh"
  PATTERN "*.inl"
)
configure_package_config_file(
  cmake/cuda-coop-tile-config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/cuda-coop-tile-config.cmake"
  INSTALL_DESTINATION "${CUDA_COOP_TILE_INSTALL_CMAKE_DIR}"
  PATH_VARS
    CUDA_COOP_TILE_INSTALL_INCLUDE_DIR
    CUDA_COOP_TILE_INSTALL_LIBRARY_DIR
)
install(
  FILES
    "${CMAKE_CURRENT_BINARY_DIR}/cuda-coop-tile-config.cmake"
    cmake/cuda-coop-tile-link.cmake
  DESTINATION "${CUDA_COOP_TILE_INSTALL_CMAKE_DIR}"
)
configure_file(
  cmake/install-provider-manifest.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/install-provider-manifest.cmake"
  @ONLY
)
install(SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/install-provider-manifest.cmake")

if (CUDA_COOP_TILE_BUILD_TESTS)
  enable_language(CUDA)
  enable_testing()
  add_subdirectory(test)
endif()
set(CUDA_COOP_TILE_INTERFACE_FILE "${_interface}" PARENT_SCOPE)
set(CUDA_COOP_TILE_LIBRARY_FILE "${_tilelib}" PARENT_SCOPE)
