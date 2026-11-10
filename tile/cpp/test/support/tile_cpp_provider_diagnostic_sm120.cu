//===----------------------------------------------------------------------===//
//
// Part of the CUDA Core Compute Libraries, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/coop/tile.cuh>

namespace ct = ::cuda::tiles;

inline constexpr size_t tile_size = 256;

// Use the canonical declaration with a contradictory dtype token, so the
// compiler admits the call and the provider reports inconsistent metadata.
// Deliberately malformed ABI declarations fail before provider discovery.
__tile_global__ void invalid_sum_dtype_kernel(const int* input, int* output)
{
  auto input_span  = ct::tensor_span{input, ct::shape<tile_size>{}};
  auto output_span = ct::tensor_span{output, ct::shape<tile_size>{}};
  auto input_view  = ct::partition_view{input_span, ct::shape<tile_size>{}};
  auto output_view = ct::partition_view{output_span, ct::shape<tile_size>{}};
  auto values      = input_view.load(0);
  auto result      = ::cuda::coop::tile::detail::sum_foreign<tile_size, int, "warp_reductions", "f32">(values);
  output_view.store(result, 0);
}
