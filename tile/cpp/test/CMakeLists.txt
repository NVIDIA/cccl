# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

foreach (_test)
  add_executable("cccl.tile.${_test}" "support/tile_cpp_${_test}.cu")
  cuda_coop_tile_link("cccl.tile.${_test}")
  add_test(NAME "cccl.tile.${_test}" COMMAND "cccl.tile.${_test}")
endforeach()
