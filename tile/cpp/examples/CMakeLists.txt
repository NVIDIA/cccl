# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.25.2)
project(tile_block_scan LANGUAGES CXX CUDA)

find_package(cuda-coop-tile CONFIG REQUIRED)
add_executable(block_scan block_scan.cu)
cuda_coop_tile_link(block_scan)
