//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/tuple>
#include <cuda/std/type_traits>

namespace cuda::experimental::stf::reserved
{

template <typename shape_t, typename static_types_tup, typename F>
__device__ void jit_loop(F f, static_types_tup args)
{
  static_assert(::cuda::std::is_empty_v<shape_t>, "jit_loop requires an empty shape type");

  size_t i          = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t step = blockDim.x * gridDim.x;
  const size_t n    = shape_t::size();

  auto explode_args = [&](auto&&... data) {
    auto const explode_coords = [&](auto&&... coords) {
      f(coords..., data...);
    };
    // For every linearized index in the shape
    for (; i < n; i += step)
    {
      ::cuda::std::apply(explode_coords, shape_t::index_to_coords(i));
    }
  };
  ::cuda::std::apply(explode_args, ::std::move(args));
}

} // namespace cuda::experimental::stf::reserved
