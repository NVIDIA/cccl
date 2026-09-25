//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 *
 * @brief Bundles: grouped logical data used as a single dependency
 *
 * Checks that a bundle dependency expands to per-field dependencies, that the
 * user function receives one tuple of views per bundle, that constant fields
 * are clamped to read access (const views) in every spelling, that fields
 * remain usable as bare logical data concurrently with bundle use, and that
 * all of this holds on both the stream and graph backends.
 */

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

using test_bundle = bundle<field<slice<double>>, field<slice<int>, constant>>;

__global__ void bundle_kernel(slice<const double> v, slice<const int> ix, slice<double> out)
{
  for (size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < out.size(); i += blockDim.x * gridDim.x)
  {
    out(i) += 0.0 * (v(i) + ix(i));
  }
}

// The rw view must keep mutable fields mutable and const-qualify constant fields
template <typename BundleView>
__host__ __device__ void check_rw_view_types()
{
  static_assert(::cuda::std::is_same_v<::cuda::std::tuple_element_t<0, BundleView>, slice<double>>,
                "mutable field must stay mutable in an rw bundle view");
  static_assert(::cuda::std::is_same_v<::cuda::std::tuple_element_t<1, BundleView>, slice<const int>>,
                "constant field must be const in an rw bundle view");
}

void run(context& ctx)
{
  constexpr size_t N = 64;

  ::std::vector<double> vals(N, 2.0);
  ::std::vector<int> idx(N);
  for (size_t i = 0; i < N; i++)
  {
    idx[i] = static_cast<int>(i);
  }

  auto lv = ctx.logical_data(&vals[0], N);
  auto li = ctx.logical_data(&idx[0], N);
  auto lo = ctx.logical_data(shape_of<slice<double>>(N));

  // Adopting constructor: no context needed, handles are shared
  test_bundle B(lv, li);
  B.set_symbol("B");

  // Fields remain first-class logical data
  static_assert(test_bundle::n_fields == 2);

  // parallel_for with an rw bundle dep: one tuple argument
  ctx.parallel_for(lo.shape(), B.rw(), lo.write())->*[] __device__(size_t i, auto b, auto out) {
    check_rw_view_types<decltype(b)>();
    out(i) = ::cuda::std::get<0>(b)(i) * ::cuda::std::get<1>(b)(i);
    ::cuda::std::get<0>(b)(i) += 1.0;
  };

  // task with a read bundle dep: all views const
  ctx.task(B.read(), lo.rw())->*[](cudaStream_t, auto b, auto) {
    static_assert(::cuda::std::is_same_v<::cuda::std::tuple_element_t<0, decltype(b)>, slice<const double>>,
                  "read bundle view must const-qualify mutable fields");
    static_assert(::cuda::std::is_same_v<::cuda::std::tuple_element_t<1, decltype(b)>, slice<const int>>,
                  "read bundle view must const-qualify constant fields");
  };

  // Mixed use: bundle dep and a bare dep on one of its own fields in the same task
  ctx.parallel_for(lo.shape(), B.read(), lv.rw())->*[] __device__(size_t i, auto b, auto v2) {
    v2(i) = ::cuda::std::get<0>(b)(i);
  };

  // token dependency mixed with a bundle dependency: the token produces no
  // lambda argument, and the bundle grouping must stay aligned
  auto tok = ctx.token();
  ctx.parallel_for(lo.shape(), tok.rw(), B.read(), lo.rw())->*[] __device__(size_t i, auto b, auto out) {
    out(i) += 0.0 * ::cuda::std::get<0>(b)(i);
  };

  // launch with a bundle dependency
  ctx.launch(B.read(), lo.rw())->*[] __device__(auto th, auto b, auto out) {
    for (size_t i = th.rank(); i < out.size(); i += th.size())
    {
      out(i) += 0.0 * ::cuda::std::get<1>(b)(i);
    }
  };

  // cuda_kernel and cuda_kernel_chain with a bundle dependency
  ctx.cuda_kernel(B.read(), lo.rw())->*[](auto b, auto out) {
    return cuda_kernel_desc{bundle_kernel, 8, 32, 0, ::cuda::std::get<0>(b), ::cuda::std::get<1>(b), out};
  };

  ctx.cuda_kernel_chain(B.read(), lo.rw())->*[](auto b, auto out) {
    return ::std::vector<cuda_kernel_desc>{
      {bundle_kernel, 8, 32, 0, ::cuda::std::get<0>(b), ::cuda::std::get<1>(b), out},
      {bundle_kernel, 8, 32, 0, ::cuda::std::get<0>(b), ::cuda::std::get<1>(b), out}};
  };

  // host_launch with structured bindings on the bundle view; verify the results
  ctx.host_launch(B.read(), lo.read())->*[](auto b, auto out) {
    auto& [v, ix] = b;
    for (size_t i = 0; i < N; i++)
    {
      EXPECT(out(i) == 2.0 * ix(i));
      EXPECT(v(i) == 3.0);
    }
  };

  ctx.finalize();
}

int main()
{
  // Creating constructor: fresh logical data from shapes, fields exposed
  {
    context ctx;
    bundle<field<slice<double>>, field<slice<int>>> C(ctx, shape_of<slice<double>>(16), shape_of<slice<int>>(16));
    ctx.parallel_for(C.get_field<0>().shape(), C.write())->*[] __device__(size_t i, auto c) {
      ::cuda::std::get<0>(c)(i) = 1.0;
      ::cuda::std::get<1>(c)(i) = 2;
    };
    // Bare-leaf consumption of a bundle-created field
    ctx.host_launch(C.get_field<1>().read())->*[](auto ci) {
      EXPECT(ci(0) == 2);
    };
    ctx.finalize();
  }

  {
    context ctx;
    run(ctx);
  }

  {
    context ctx = graph_ctx();
    run(ctx);
  }

  return 0;
}
