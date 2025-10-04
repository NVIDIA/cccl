//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

//! \file
//! \brief DOT algorithm

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

// Note that a and b might be the same logical data
template <typename ctx_t, typename T>
void DOT(ctx_t& ctx,
         stackable_logical_data<slice<T>>& a,
         stackable_logical_data<slice<T>>& b,
         stackable_logical_data<scalar_view<T>>& res)
{
  ctx.parallel_for(a.shape(), a.read(), b.read(), res.reduce(reducer::sum<T>{})).set_symbol("DOT")->*
    [] __device__(size_t i, auto da, auto db, T& dres) {
      dres += da(i) * db(i);
    };
};
