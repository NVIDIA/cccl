# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

foreach (
  _test
  scan_reduce
  scan_prefix
  reduce_axes
  radix_sort_sm120
  radix_rank_sm120
  exchange_smoke
  histogram_smoke
  histogram_256_smoke
  discontinuity_flags_smoke
  adjacent_shuffle_sm120
  merge_sort
  merge_sort_keys_smoke
  run_length_decode_sm120
)
  add_executable("cccl.tile.${_test}" "support/tile_cpp_${_test}.cu")
  cuda_coop_tile_link("cccl.tile.${_test}")
  add_test(NAME "cccl.tile.${_test}" COMMAND "cccl.tile.${_test}")
endforeach()
