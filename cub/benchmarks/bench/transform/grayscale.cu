// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// %RANGE% TUNE_BIF_BIAS bif -16:16:4
// %RANGE% TUNE_ALGORITHM alg 0:4:1
// %RANGE% TUNE_THREADS tpb 128:1024:128

// those parameters only apply if TUNE_ALGORITHM == 1 (vectorized)
// %RANGE% TUNE_VEC_SIZE_POW2 vsp2 1:6:1
// %RANGE% TUNE_VECTORS_PER_THREAD vpt 1:4:1

#if !TUNE_BASE && TUNE_ALGORITHM != 1 && (TUNE_VEC_SIZE_POW2 != 1 || TUNE_VECTORS_PER_THREAD != 1)
#  error "Non-vectorized algorithms require vector size and vectors per thread to be 1 since they ignore the parameters"
#endif // !TUNE_BASE && TUNE_ALGORITHM != 1 && (TUNE_VEC_SIZE_POW2 != 1 || TUNE_VECTORS_PER_THREAD != 1)

#include "common.h"

template <typename T>
struct rgb_t
{
  T r;
  T g;
  T b;

  __device__ T grayscale() const
  {
    static constexpr T w_r(0.2989);
    static constexpr T w_g(0.587);
    static constexpr T w_b(0.114);

    return w_r * r + w_g * g + w_b * b;
  }
};

template <typename T>
struct transform_op_t
{
  __device__ T operator()(rgb_t<T> pixel) const
  {
    return pixel.grayscale();
  }
};

template <typename T, typename OffsetT>
static void grayscale(nvbench::state& state, nvbench::type_list<T, OffsetT>)
try
{
  using pixel_t = rgb_t<T>;
  const auto n  = state.get_int64("Elements{io}");
  if (sizeof(OffsetT) == 4 && n > std::numeric_limits<OffsetT>::max())
  {
    state.skip("Skipping: input size exceeds 32-bit offset type capacity.");
    return;
  }

  // Generate random RGB data by creating separate R, G, B vectors and combining them
  thrust::device_vector<T> r_data = generate(n);
  thrust::device_vector<T> g_data = generate(n);
  thrust::device_vector<T> b_data = generate(n);

  thrust::device_vector<pixel_t> input(n, thrust::no_init);
  thrust::transform(
    thrust::make_zip_iterator(r_data.begin(), g_data.begin(), b_data.begin()),
    thrust::make_zip_iterator(r_data.end(), g_data.end(), b_data.end()),
    input.begin(),
    thrust::make_zip_function([] __device__(T r, T g, T b) {
      return pixel_t{r, g, b};
    }));

  thrust::device_vector<T> output(n, thrust::no_init);

  state.add_element_count(n);
  state.add_global_memory_reads<pixel_t>(n);
  state.add_global_memory_writes<T>(n);

  bench_transform(state, cuda::std::tuple{input.begin()}, output.begin(), static_cast<OffsetT>(n), transform_op_t<T>{});
}
catch (const std::bad_alloc&)
{
  state.skip("Skipping: out of memory.");
}

#ifdef TUNE_T
using value_types = nvbench::type_list<TUNE_T>;
#else
using value_types = nvbench::type_list<float, double>;
#endif

NVBENCH_BENCH_TYPES(grayscale, NVBENCH_TYPE_AXES(value_types, offset_types))
  .set_name("grayscale")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 32, 4));
