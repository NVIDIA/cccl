// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Tile variant of the grayscale transform bench. Unlike the base bench (a single rgb_t<T> struct
// input), this uses three separate R/G/B streams so the inputs are plain element types the tile path
// can vectorize. The named rgb_to_y op registers a tile_operator substitute (gated). This file
// disappears once tile dispatch is fully transparent.

#include "../common.h"

#if _CCCL_CUB_TILE_TRANSFORM_DISPATCH_ENABLED()
#  include <cuda_tile.h>
#endif

struct rgb_to_y
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class R, class G, class B>
  _CCCL_API auto operator()(R r, G g, B b) const
  {
    constexpr float w_r = 0.2989f;
    constexpr float w_g = 0.587f;
    constexpr float w_b = 0.114f;
    return w_r * r + w_g * g + w_b * b;
  }
};

#if _CCCL_CUB_TILE_TRANSFORM_DISPATCH_ENABLED()
CUB_NAMESPACE_BEGIN
namespace detail::transform::tile
{
template <class T>
inline constexpr bool tile_eligible_v<rgb_to_y, T, 3> = true;
template <>
struct tile_operator<rgb_to_y>
{
  using type = rgb_to_y;
};
} // namespace detail::transform::tile
CUB_NAMESPACE_END
#endif // _CCCL_CUB_TILE_TRANSFORM_DISPATCH_ENABLED()

#ifdef TUNE_T
using value_types = nvbench::type_list<TUNE_T>;
#else
using value_types = nvbench::type_list<nvbench::float32_t, nvbench::float64_t>;
#endif

template <typename T>
static void grayscale(nvbench::state& state, nvbench::type_list<T>)
try
{
  const auto n = state.get_int64("Elements{io}");

  thrust::device_vector<T> r = generate(n);
  thrust::device_vector<T> g = generate(n);
  thrust::device_vector<T> b = generate(n);
  thrust::device_vector<T> out(n, thrust::no_init);

  state.add_element_count(n);
  state.add_global_memory_reads<T>(3 * n); // matches the base bench's rgb_t<T> = 3 * sizeof(T)
  state.add_global_memory_writes<T>(n);
  bench_transform(state, cuda::std::tuple{r.begin(), g.begin(), b.begin()}, out.begin(), n, rgb_to_y{});
}
catch (const std::bad_alloc&)
{
  state.skip("Skipping: out of memory.");
}

NVBENCH_BENCH_TYPES(grayscale, NVBENCH_TYPE_AXES(value_types))
  .set_name("tile_grayscale")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 32, 4));
