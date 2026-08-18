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
 * @brief Frozen access to a composite place with replicated axes: freeze at
 * the replicated place (read), then get() at each member place -- one
 * composite instance per replicated coordinate, each striped over its
 * fiber's bound-axes places. The interop pattern for handing per-replica
 * pointers out of STF. Also drives the stackable member walk through a
 * composite-replicated place.
 */

#include <cuda/experimental/stf.cuh>

#include <vector>

using namespace cuda::experimental::stf;

int main()
{
  const size_t n = 1 << 20;

  // ---- freeze + get at the member places: (2, 2) grid, tensor blocked
  // over axis 0, one copy per coordinate of axis 1
  {
    stream_ctx ctx;
    cudaStream_t stream = ctx.pick_stream();

    ::std::vector<exec_place> places(4, exec_place::current_device());
    auto grid = make_grid(mv(places), dim4(2, 2));
    auto part =
      make_partition_descriptor(dim4(n), {dim_spec{dim_policy::blocked, 0, 0}}, dim4(2, 2), /*replicated_axes=*/0x2);
    auto cdp = make_composite_data_place(grid, part);

    EXPECT(cdp.is_replicated());
    EXPECT(cdp.instance_count() == 2);

    auto lX = ctx.logical_data(shape_of<slice<double>>(n)).set_symbol("X");
    ctx.parallel_for(lX.shape(), lX.write())->*[] __device__(size_t i, auto x) {
      x(i) = static_cast<double>(i % 128);
    };

    // Freeze at the replicated place, get at the members (a get at the
    // replicated place itself is not a thing: the fan-out is per member)
    auto fx = ctx.freeze(lX, access_mode::read, cdp);

    ::std::vector<double> host(n);
    for (size_t r = 0; r < cdp.instance_count(); r++)
    {
      const data_place member = cdp.member(r);
      EXPECT(member.is_composite());
      EXPECT(!member.is_replicated());

      auto s = fx.get(member, stream);
      cuda_safe_call(cudaMemcpyAsync(host.data(), s.data_handle(), n * sizeof(double), cudaMemcpyDeviceToHost, stream));
      cuda_safe_call(cudaStreamSynchronize(stream));
      for (size_t i = 0; i < n; i++)
      {
        EXPECT(host[i] == static_cast<double>(i % 128));
      }
    }
    fx.unfreeze(stream);
    ctx.finalize();
    printf("composite frozen get: member instances OK\n");
  }

  // ---- stackable member walk through a composite-replicated place: the
  // push at the replicated place imports every member instance, so the
  // repeated reads resolve in the nested context without copies (1-D grid:
  // full replication, members degenerate to the affine device places)
#if _CCCL_CTK_AT_LEAST(12, 4)
  {
    stackable_ctx ctx;
    const size_t nplaces = 2;
    auto grid            = exec_place::repeat(exec_place::current_device(), nplaces);
    auto part            = make_partition_descriptor(dim4(n), {dim_spec{}}, dim4(nplaces), /*replicated_axes=*/0x1);
    auto cdp             = make_composite_data_place(grid, part);
    EXPECT(cdp.is_replicated());

    auto lin  = ctx.logical_data(shape_of<slice<double>>(n)).set_symbol("cin");
    auto lacc = ctx.logical_data(shape_of<slice<double>>(n)).set_symbol("cacc");
    ctx.parallel_for(lin.shape(), lin.write(), lacc.write())->*[] __device__(size_t i, auto in, auto acc) {
      in(i)  = static_cast<double>(i % 64);
      acc(i) = 0.0;
    };

    const size_t iters = 6;
    {
      auto rg = ctx.repeat_graph_scope(iters);
      lin.push(access_mode::read, cdp);
      lacc.push(access_mode::rw, data_place::composite(blocked_partition(), grid));
      ctx.parallel_for(blocked_partition(), grid, lin.shape(), lin.read(cdp), lacc.rw())
          ->*[] __device__(size_t i, auto in, auto acc) {
                acc(i) += in(i);
              };
    }

    ctx.host_launch(lacc.read())->*[&](auto acc) {
      for (size_t i = 0; i < n; i++)
      {
        EXPECT(acc(i) == static_cast<double>(iters) * static_cast<double>(i % 64));
      }
    };
    ctx.finalize();
    printf("composite frozen get: stackable member walk OK\n");
  }
#endif // _CCCL_CTK_AT_LEAST(12, 4)

  printf("replicated_composite_frozen_get: all checks passed\n");
  return 0;
}
