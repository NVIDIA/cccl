//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cub/agent/single_pass_scan_operators.cuh>
#include <cub/util_arch.cuh>

// cub::ReduceByKeyScanTileState has two layouts, and the driver launcher passes the tile state to the
// JIT-compiled kernel *by value*, so the host type has to match the kernel's parameter size:
//
//   SingleWord == true   one pointer; the per-tile word is the next power of two above
//                        sizeof(ValueT) + sizeof(KeyT) + 1
//   SingleWord == false  three pointers, which is exactly scan_tile_state's layout
//
// The multi-word form therefore reuses scan_tile_state. Only the single-word form needs a new type,
// and its word size is a template parameter so AllocationSize stays exact without widening the struct.
template <int TxnWordSize>
struct reduce_by_key_single_word_tile_state
{
  void* d_tile_descriptors;

  cudaError_t Init(int, void* d_temp_storage, size_t)
  {
    d_tile_descriptors = d_temp_storage;
    return cudaSuccess;
  }

  cudaError_t AllocationSize(int num_tiles, size_t& temp_storage_bytes) const
  {
    temp_storage_bytes = static_cast<size_t>(num_tiles + cub::detail::warp_threads) * TxnWordSize;
    return cudaSuccess;
  }
};

// Mirrors ReduceByKeyScanTileState's own TxnWord sizing: next power of two above the pair size plus one.
constexpr int reduce_by_key_txn_word_size(int key_size, int value_size)
{
  int size = 4;
  while (size < key_size + value_size + 1)
  {
    size <<= 1;
  }
  return size;
}

// The primitive check is approximated the same way the scan backend approximates it: every type the C
// API currently accepts is primitive or trivially copyable, so only the size test decides.
constexpr bool reduce_by_key_is_single_word(int key_size, int value_size)
{
  return key_size + value_size < cub::detail::largest_atomic_message_size;
}
