// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

// This benchmark tests overlapping memory regions for reading and is compute intensive

template <typename OffsetT>
static void compare_complex(nvbench::state& state, nvbench::type_list<OffsetT>)
try
{
  const auto n = state.get_int64("Elements{io}");
  if (sizeof(OffsetT) == 4 && n > std::numeric_limits<OffsetT>::max())
  {
    state.skip("Skipping: input size exceeds 32-bit offset type capacity.");
    return;
  }

  thrust::device_vector<complex> in = generate(n);
  thrust::device_vector<bool> out(n - 1);

  state.add_element_count(n);
  state.add_global_memory_reads<complex>(n);
  state.add_global_memory_writes<bool>(n);

  // the complex comparison needs lots of compute and transform reads from overlapping input
  using compare_op = less_t;
  bench_transform(
    state, ::cuda::std::tuple{in.begin(), in.begin() + 1}, out.begin(), static_cast<OffsetT>(n) - 1, compare_op{});
}
catch (const std::bad_alloc&)
{
  state.skip("Skipping: out of memory.");
}

// TODO(bgruber): hardcode OffsetT?
NVBENCH_BENCH_TYPES(compare_complex, NVBENCH_TYPE_AXES(offset_types))
  .set_name("compare_complex")
  .set_type_axes_names({"OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 32, 4));
