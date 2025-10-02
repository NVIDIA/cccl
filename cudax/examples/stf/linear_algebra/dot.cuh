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

template <typename T>
using vector_t = stackable_logical_data<slice<T>>;

template <typename T>
using scalar_t = stackable_logical_data<scalar_view<T>>;

template <typename T = double>
struct csr_matrix
{
  csr_matrix(stackable_logical_data<slice<T>> _val_handle,
             stackable_logical_data<slice<size_t>> _row_handle,
             stackable_logical_data<slice<size_t>> _col_handle)
      : val_handle(mv(_val_handle))
      , row_handle(mv(_row_handle))
      , col_handle(mv(_col_handle))
  {}

  /* Description of the CSR */
  mutable stackable_logical_data<slice<T>> val_handle;
  mutable stackable_logical_data<slice<size_t>> row_handle;
  mutable stackable_logical_data<slice<size_t>> col_handle;
};



// Note that a and b might be the same logical data
template <typename ctx_t, typename T>
void DOT(ctx_t& ctx,
         vector_t<T>& a,
         vector_t<T>& b,
         scalar_t<T>& res)
{
  ctx.parallel_for(a.shape(), a.read(), b.read(), res.reduce(reducer::sum<T>{})).set_symbol("DOT")->*
    [] __device__(size_t i, auto da, auto db, T& dres) {
      dres += da(i) * db(i);
    };
};

template <typename ctx_t, typename T>
void SPMV(ctx_t& ctx, csr_matrix<T>& a, vector_t<T>& x, vector_t<T>& y)
{
  ctx.parallel_for(y.shape(), a.val_handle.read(), a.col_handle.read(), a.row_handle.read(), x.read(), y.write())
      .set_symbol("SPMV")
      ->*[] _CCCL_DEVICE(size_t row, auto da_val, auto da_col, auto da_row, auto dx, auto dy) {
            int row_start = da_row(row);
            int row_end   = da_row(row + 1);

            double sum = 0.0;
            for (int elt = row_start; elt < row_end; elt++)
            {
              sum += da_val(elt) * dx(da_col(elt));
            }

            dy(row) = sum;
          };
}

