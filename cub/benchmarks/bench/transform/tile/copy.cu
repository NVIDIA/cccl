// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Pure copy (identity transform) -- measures plain load/store bandwidth through the tile
// load_masked/store_masked path. The identity op registers a tile_operator substitute (gated); under
// --enable-tile + CCCL_ENABLE_EXPERIMENTAL_TILE_TRANSFORM_DISPATCH the dispatch hook routes it to the tile kernel,
// otherwise it falls through to CUB's standard transform. This file disappears once tile dispatch is
// fully transparent.

#include "../common.h"

#if _CCCL_CUB_TILE_TRANSFORM_DISPATCH_ENABLED()
#  include <cuda_tile.h>
#endif

struct identity
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class T>
  _CCCL_API auto operator()(T v) const
  {
    return v;
  }
};

#if _CCCL_CUB_TILE_TRANSFORM_DISPATCH_ENABLED()
CUB_NAMESPACE_BEGIN
namespace detail::transform::tile
{
template <class T>
inline constexpr bool tile_eligible_v<identity, T, 1> = true;
template <>
struct tile_operator<identity>
{
  using type = identity;
};
} // namespace detail::transform::tile
CUB_NAMESPACE_END
#endif // _CCCL_CUB_TILE_TRANSFORM_DISPATCH_ENABLED()

#ifdef TUNE_T
using element_types = nvbench::type_list<TUNE_T>;
#else
using element_types = nvbench::type_list<nvbench::int8_t, nvbench::int16_t, nvbench::int32_t, nvbench::float64_t>;
#endif

template <typename T>
static void copy(nvbench::state& state, nvbench::type_list<T>)
try
{
  const auto n = state.get_int64("Elements{io}");

  thrust::device_vector<T> in = generate(n);
  thrust::device_vector<T> out(n, thrust::no_init);

  state.add_element_count(n);
  state.add_global_memory_reads<T>(n);
  state.add_global_memory_writes<T>(n);
  bench_transform(state, cuda::std::tuple{in.begin()}, out.begin(), n, identity{});
}
catch (const std::bad_alloc&)
{
  state.skip("Skipping: out of memory.");
}

NVBENCH_BENCH_TYPES(copy, NVBENCH_TYPE_AXES(element_types))
  .set_name("tile_copy")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 32, 4));
