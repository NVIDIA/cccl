// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/shuffle.h>

#include "nvbench_helper.cuh"

template <typename T>
static void basic(nvbench::state& state, nvbench::type_list<T>)
{
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements"));

  thrust::device_vector<T> data(elements);

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements);
  state.add_global_memory_writes<T>(elements);

  auto do_engine = [&](auto&& engine_constructor) {
    caching_allocator_t alloc;
    state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch | nvbench::exec_tag::sync,
               [&](nvbench::launch& launch) {
                 thrust::shuffle(policy(alloc, launch), data.begin(), data.end(), engine_constructor());
               });
  };

  const auto rng_engine = state.get_string("Engine");
  if (rng_engine == "minstd")
  {
    do_engine([] {
      return thrust::random::minstd_rand{};
    });
  }
  else if (rng_engine == "ranlux24")
  {
    do_engine([] {
      return thrust::random::ranlux24{};
    });
  }
  else if (rng_engine == "ranlux48")
  {
    do_engine([] {
      return thrust::random::ranlux48{};
    });
  }
  else if (rng_engine == "taus88")
  {
    do_engine([] {
      return thrust::random::taus88{};
    });
  }
}

using types =
  nvbench::type_list<int8_t,
                     int16_t,
                     int32_t,
                     int64_t
#if NVBENCH_HELPER_HAS_I128
                     ,
                     int128_t
#endif
                     >;

NVBENCH_BENCH_TYPES(basic, NVBENCH_TYPE_AXES(types))
  .set_name("base")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4))
  .add_string_axis("Engine", {"minstd", "ranlux24", "ranlux48", "taus88"});
