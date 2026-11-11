//===----------------------------------------------------------------------===//
//
// Part of the CUDA Core Compute Libraries, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// This is CCCL's library-owned source of truth for cuda.coop.tile and
// cuda::coop::tile foreign declarations. The TileLib packages these 21
// preview$func schemas with the linkage-provider implementation.

cuda_tile.module @cuda_coop_tile_interface {
  cuda_tile.preview$func extern @cuda_coop_tile_scan<
      [N: index,
       T: !cuda_tile.preview$Datatype,
       Mode: !cuda_tile.preview$string,
       Algorithm: !cuda_tile.preview$string,
       ScanOp: !cuda_tile.preview$string,
       InitialPresent: index,
       PrefixPresent: index,
       DTypeToken: !cuda_tile.preview$string]>(
      %input: !cuda_tile.tile<[N]x!cuda_tile.preview$sym_t<T>>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>},
      %runtime_initial: !cuda_tile.tile<!cuda_tile.preview$sym_t<T>>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>},
      %runtime_prefix: !cuda_tile.tile<!cuda_tile.preview$sym_t<T>>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>},
      %output: !cuda_tile.tile<[N]x!cuda_tile.preview$sym_t<T>>
        {cuda_tile.preview$access = #cuda_tile.preview$access<out>})
    -> !cuda_tile.tile<[N]x!cuda_tile.preview$sym_t<T>>

  cuda_tile.preview$func extern @cuda_coop_tile_sum<
      [N: index,
       T: !cuda_tile.preview$Datatype,
       Algorithm: !cuda_tile.preview$string,
       DTypeToken: !cuda_tile.preview$string]>(
      %input: !cuda_tile.tile<[N]x!cuda_tile.preview$sym_t<T>>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>},
      %output: !cuda_tile.tile<[N]x!cuda_tile.preview$sym_t<T>>
        {cuda_tile.preview$access = #cuda_tile.preview$access<out>})
    -> !cuda_tile.tile<[N]x!cuda_tile.preview$sym_t<T>>

  cuda_tile.preview$func extern @cuda_coop_tile_reduce<
      [N: index,
       T: !cuda_tile.preview$Datatype,
       Algorithm: !cuda_tile.preview$string,
       BinaryOp: !cuda_tile.preview$string,
       DTypeToken: !cuda_tile.preview$string]>(
      %input: !cuda_tile.tile<[N]x!cuda_tile.preview$sym_t<T>>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>},
      %output: !cuda_tile.tile<[N]x!cuda_tile.preview$sym_t<T>>
        {cuda_tile.preview$access = #cuda_tile.preview$access<out>})
    -> !cuda_tile.tile<[N]x!cuda_tile.preview$sym_t<T>>

  cuda_tile.preview$func extern @cuda_coop_tile_sum_batched<
      [N: index,
       T: !cuda_tile.preview$Datatype,
       Algorithm: !cuda_tile.preview$string,
       DTypeToken: !cuda_tile.preview$string,
       Batches: index]>(
      %input: !cuda_tile.tile<[N]x!cuda_tile.preview$sym_t<T>>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>},
      %output: !cuda_tile.tile<[Batches]x!cuda_tile.preview$sym_t<T>>
        {cuda_tile.preview$access = #cuda_tile.preview$access<out>})
    -> !cuda_tile.tile<[Batches]x!cuda_tile.preview$sym_t<T>>

  cuda_tile.preview$func extern @cuda_coop_tile_reduce_batched<
      [N: index,
       T: !cuda_tile.preview$Datatype,
       Algorithm: !cuda_tile.preview$string,
       BinaryOp: !cuda_tile.preview$string,
       DTypeToken: !cuda_tile.preview$string,
       Batches: index]>(
      %input: !cuda_tile.tile<[N]x!cuda_tile.preview$sym_t<T>>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>},
      %output: !cuda_tile.tile<[Batches]x!cuda_tile.preview$sym_t<T>>
        {cuda_tile.preview$access = #cuda_tile.preview$access<out>})
    -> !cuda_tile.tile<[Batches]x!cuda_tile.preview$sym_t<T>>

  cuda_tile.preview$func extern @cuda_coop_tile_histogram<
      [N: index,
       Bins: index,
       DTypeToken: !cuda_tile.preview$string]>(
      %input: !cuda_tile.tile<[N]xi8>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>},
      %output: !cuda_tile.tile<[Bins]xi32>
        {cuda_tile.preview$access = #cuda_tile.preview$access<out>})
    -> !cuda_tile.tile<[Bins]xi32>

  cuda_tile.preview$func extern @cuda_coop_tile_histogram_256<
      [N: index,
       DTypeToken: !cuda_tile.preview$string]>(
      %input: !cuda_tile.tile<[N]xi8>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>},
      %output: !cuda_tile.tile<256xi32>
        {cuda_tile.preview$access = #cuda_tile.preview$access<out>})
    -> !cuda_tile.tile<256xi32>

  cuda_tile.preview$func extern @cuda_coop_tile_run_length_decode<
      [RunN: index,
       DecodedN: index,
       ItemT: !cuda_tile.preview$Datatype,
       LengthT: !cuda_tile.preview$Datatype,
       ItemDTypeToken: !cuda_tile.preview$string,
       LengthDTypeToken: !cuda_tile.preview$string]>(
      %run_values: !cuda_tile.tile<[RunN]x!cuda_tile.preview$sym_t<ItemT>>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>},
      %run_lengths: !cuda_tile.tile<[RunN]x!cuda_tile.preview$sym_t<LengthT>>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>},
      %decoded_window_offset: !cuda_tile.tile<!cuda_tile.preview$sym_t<LengthT>>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>},
      %decoded_items: !cuda_tile.tile<[DecodedN]x!cuda_tile.preview$sym_t<ItemT>>
        {cuda_tile.preview$access = #cuda_tile.preview$access<out>})
    -> !cuda_tile.tile<[DecodedN]x!cuda_tile.preview$sym_t<ItemT>>

  cuda_tile.preview$func extern
      @cuda_coop_tile_run_length_decode_with_offsets<
      [RunN: index,
       DecodedN: index,
       ItemT: !cuda_tile.preview$Datatype,
       LengthT: !cuda_tile.preview$Datatype,
       ItemDTypeToken: !cuda_tile.preview$string,
       LengthDTypeToken: !cuda_tile.preview$string]>(
      %run_values: !cuda_tile.tile<[RunN]x!cuda_tile.preview$sym_t<ItemT>>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>},
      %run_lengths: !cuda_tile.tile<[RunN]x!cuda_tile.preview$sym_t<LengthT>>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>},
      %decoded_window_offset: !cuda_tile.tile<!cuda_tile.preview$sym_t<LengthT>>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>},
      %decoded_items: !cuda_tile.tile<[DecodedN]x!cuda_tile.preview$sym_t<ItemT>>
        {cuda_tile.preview$access = #cuda_tile.preview$access<out>},
      %relative_offsets:
          !cuda_tile.tile<[DecodedN]x!cuda_tile.preview$sym_t<LengthT>>
        {cuda_tile.preview$access = #cuda_tile.preview$access<out>})
    -> (!cuda_tile.tile<[DecodedN]x!cuda_tile.preview$sym_t<ItemT>>,
        !cuda_tile.tile<[DecodedN]x!cuda_tile.preview$sym_t<LengthT>>)

  cuda_tile.preview$func extern @cuda_coop_tile_discontinuity_flags<
      [N: index,
       T: !cuda_tile.preview$Datatype,
       Mode: !cuda_tile.preview$string,
       DTypeToken: !cuda_tile.preview$string]>(
      %input: !cuda_tile.tile<[N]x!cuda_tile.preview$sym_t<T>>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>},
      %output: !cuda_tile.tile<[N]xi32>
        {cuda_tile.preview$access = #cuda_tile.preview$access<out>})
    -> !cuda_tile.tile<[N]xi32>

  cuda_tile.preview$func extern
      @cuda_coop_tile_discontinuity_heads_and_tails<
      [N: index,
       T: !cuda_tile.preview$Datatype,
       Mode: !cuda_tile.preview$string,
       DTypeToken: !cuda_tile.preview$string]>(
      %input: !cuda_tile.tile<[N]x!cuda_tile.preview$sym_t<T>>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>},
      %heads: !cuda_tile.tile<[N]xi32>
        {cuda_tile.preview$access = #cuda_tile.preview$access<out>},
      %tails: !cuda_tile.tile<[N]xi32>
        {cuda_tile.preview$access = #cuda_tile.preview$access<out>})
    -> (!cuda_tile.tile<[N]xi32>, !cuda_tile.tile<[N]xi32>)

  cuda_tile.preview$func extern @cuda_coop_tile_adjacent_difference<
      [N: index,
       T: !cuda_tile.preview$Datatype,
       Mode: !cuda_tile.preview$string,
       DTypeToken: !cuda_tile.preview$string]>(
      %input: !cuda_tile.tile<[N]x!cuda_tile.preview$sym_t<T>>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>},
      %output: !cuda_tile.tile<[N]x!cuda_tile.preview$sym_t<T>>
        {cuda_tile.preview$access = #cuda_tile.preview$access<out>})
    -> !cuda_tile.tile<[N]x!cuda_tile.preview$sym_t<T>>

  cuda_tile.preview$func extern @cuda_coop_tile_shuffle<
      [N: index,
       T: !cuda_tile.preview$Datatype,
       Mode: !cuda_tile.preview$string,
       DTypeToken: !cuda_tile.preview$string]>(
      %input: !cuda_tile.tile<[N]x!cuda_tile.preview$sym_t<T>>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>},
      %distance: !cuda_tile.tile<i32>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>},
      %output: !cuda_tile.tile<[N]x!cuda_tile.preview$sym_t<T>>
        {cuda_tile.preview$access = #cuda_tile.preview$access<out>})
    -> !cuda_tile.tile<[N]x!cuda_tile.preview$sym_t<T>>

  cuda_tile.preview$func extern @cuda_coop_tile_exchange<
      [N: index,
       T: !cuda_tile.preview$Datatype,
       BlockThreads: index,
       Mode: !cuda_tile.preview$string,
       DTypeToken: !cuda_tile.preview$string]>(
      %input_output: !cuda_tile.tile<[N]x!cuda_tile.preview$sym_t<T>>
        {cuda_tile.preview$access = #cuda_tile.preview$access<inout>})
    -> !cuda_tile.tile<[N]x!cuda_tile.preview$sym_t<T>>

  cuda_tile.preview$func extern @cuda_coop_tile_radix_rank<
      [N: index,
       T: !cuda_tile.preview$Datatype,
       RadixBits: index,
       Descending: index,
       MemoizeOuterScan: index,
       InnerScanAlgorithm: !cuda_tile.preview$string,
       SmemConfig: !cuda_tile.preview$string,
       DTypeToken: !cuda_tile.preview$string]>(
      %keys: !cuda_tile.tile<[N]x!cuda_tile.preview$sym_t<T>>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>},
      %begin_bit: !cuda_tile.tile<i32>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>},
      %end_bit: !cuda_tile.tile<i32>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>},
      %output: !cuda_tile.tile<[N]xi32>
        {cuda_tile.preview$access = #cuda_tile.preview$access<out>})
    -> !cuda_tile.tile<[N]xi32>

  cuda_tile.preview$func extern @cuda_coop_tile_merge_sort_keys<
      [N: index,
       T: !cuda_tile.preview$Datatype,
       Descending: index,
       PartialTile: index,
       DTypeToken: !cuda_tile.preview$string]>(
      %keys: !cuda_tile.tile<[N]x!cuda_tile.preview$sym_t<T>>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>},
      %valid_items: !cuda_tile.tile<i32>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>},
      %oob_default: !cuda_tile.tile<!cuda_tile.preview$sym_t<T>>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>},
      %output: !cuda_tile.tile<[N]x!cuda_tile.preview$sym_t<T>>
        {cuda_tile.preview$access = #cuda_tile.preview$access<out>})
    -> !cuda_tile.tile<[N]x!cuda_tile.preview$sym_t<T>>

  cuda_tile.preview$func extern @cuda_coop_tile_merge_sort_pairs<
      [N: index,
       KeyT: !cuda_tile.preview$Datatype,
       ValueT: !cuda_tile.preview$Datatype,
       Descending: index,
       PartialTile: index,
       KeyDTypeToken: !cuda_tile.preview$string,
       ValueDTypeToken: !cuda_tile.preview$string]>(
      %keys: !cuda_tile.tile<[N]x!cuda_tile.preview$sym_t<KeyT>>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>},
      %values: !cuda_tile.tile<[N]x!cuda_tile.preview$sym_t<ValueT>>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>},
      %valid_items: !cuda_tile.tile<i32>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>},
      %oob_default: !cuda_tile.tile<!cuda_tile.preview$sym_t<KeyT>>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>},
      %output_keys: !cuda_tile.tile<[N]x!cuda_tile.preview$sym_t<KeyT>>
        {cuda_tile.preview$access = #cuda_tile.preview$access<out>},
      %output_values: !cuda_tile.tile<[N]x!cuda_tile.preview$sym_t<ValueT>>
        {cuda_tile.preview$access = #cuda_tile.preview$access<out>})
    -> (!cuda_tile.tile<[N]x!cuda_tile.preview$sym_t<KeyT>>,
        !cuda_tile.tile<[N]x!cuda_tile.preview$sym_t<ValueT>>)

  cuda_tile.preview$func extern @cuda_coop_tile_radix_sort_keys<
      [N: index,
       T: !cuda_tile.preview$Datatype,
       Descending: index,
       RadixBits: index,
       MemoizeOuterScan: index,
       InnerScanAlgorithm: !cuda_tile.preview$string,
       SmemConfig: !cuda_tile.preview$string,
       DTypeToken: !cuda_tile.preview$string]>(
      %keys: !cuda_tile.tile<[N]x!cuda_tile.preview$sym_t<T>>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>},
      %begin_bit: !cuda_tile.tile<i32>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>},
      %end_bit: !cuda_tile.tile<i32>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>},
      %output: !cuda_tile.tile<[N]x!cuda_tile.preview$sym_t<T>>
        {cuda_tile.preview$access = #cuda_tile.preview$access<out>})
    -> !cuda_tile.tile<[N]x!cuda_tile.preview$sym_t<T>>

  cuda_tile.preview$func extern @cuda_coop_tile_radix_sort_pairs<
      [N: index,
       KeyT: !cuda_tile.preview$Datatype,
       ValueT: !cuda_tile.preview$Datatype,
       Descending: index,
       RadixBits: index,
       MemoizeOuterScan: index,
       InnerScanAlgorithm: !cuda_tile.preview$string,
       SmemConfig: !cuda_tile.preview$string,
       KeyDTypeToken: !cuda_tile.preview$string,
       ValueDTypeToken: !cuda_tile.preview$string]>(
      %keys: !cuda_tile.tile<[N]x!cuda_tile.preview$sym_t<KeyT>>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>},
      %values: !cuda_tile.tile<[N]x!cuda_tile.preview$sym_t<ValueT>>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>},
      %begin_bit: !cuda_tile.tile<i32>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>},
      %end_bit: !cuda_tile.tile<i32>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>},
      %output_keys: !cuda_tile.tile<[N]x!cuda_tile.preview$sym_t<KeyT>>
        {cuda_tile.preview$access = #cuda_tile.preview$access<out>},
      %output_values: !cuda_tile.tile<[N]x!cuda_tile.preview$sym_t<ValueT>>
        {cuda_tile.preview$access = #cuda_tile.preview$access<out>})
    -> (!cuda_tile.tile<[N]x!cuda_tile.preview$sym_t<KeyT>>,
        !cuda_tile.tile<[N]x!cuda_tile.preview$sym_t<ValueT>>)

  cuda_tile.preview$func extern @cuda_coop_tile_topk_keys<
      [N: index,
       T: !cuda_tile.preview$Datatype,
       Descending: index,
       FullTile: index,
       DTypeToken: !cuda_tile.preview$string]>(
      %keys: !cuda_tile.tile<[N]x!cuda_tile.preview$sym_t<T>>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>},
      %k: !cuda_tile.tile<i32>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>},
      %num_valid: !cuda_tile.tile<i32>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>},
      %begin_bit: !cuda_tile.tile<i32>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>},
      %end_bit: !cuda_tile.tile<i32>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>},
      %output: !cuda_tile.tile<[N]x!cuda_tile.preview$sym_t<T>>
        {cuda_tile.preview$access = #cuda_tile.preview$access<out>})
    -> !cuda_tile.tile<[N]x!cuda_tile.preview$sym_t<T>>

  cuda_tile.preview$func extern @cuda_coop_tile_topk_pairs<
      [N: index,
       KeyT: !cuda_tile.preview$Datatype,
       ValueT: !cuda_tile.preview$Datatype,
       Descending: index,
       FullTile: index,
       KeyDTypeToken: !cuda_tile.preview$string,
       ValueDTypeToken: !cuda_tile.preview$string]>(
      %keys: !cuda_tile.tile<[N]x!cuda_tile.preview$sym_t<KeyT>>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>},
      %values: !cuda_tile.tile<[N]x!cuda_tile.preview$sym_t<ValueT>>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>},
      %k: !cuda_tile.tile<i32>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>},
      %num_valid: !cuda_tile.tile<i32>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>},
      %begin_bit: !cuda_tile.tile<i32>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>},
      %end_bit: !cuda_tile.tile<i32>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>},
      %output_keys: !cuda_tile.tile<[N]x!cuda_tile.preview$sym_t<KeyT>>
        {cuda_tile.preview$access = #cuda_tile.preview$access<out>},
      %output_values: !cuda_tile.tile<[N]x!cuda_tile.preview$sym_t<ValueT>>
        {cuda_tile.preview$access = #cuda_tile.preview$access<out>})
    -> (!cuda_tile.tile<[N]x!cuda_tile.preview$sym_t<KeyT>>,
        !cuda_tile.tile<[N]x!cuda_tile.preview$sym_t<ValueT>>)
}
