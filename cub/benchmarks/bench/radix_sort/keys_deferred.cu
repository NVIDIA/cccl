// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_radix_sort.cuh>

#include <thrust/device_vector.h>

#include <cuda/argument>
#include <cuda/execution>
#include <cuda/std/algorithm>
#include <cuda/std/limits>
#include <cuda/std/span>

#include <cstddef>
#include <new>

#include <nvbench_helper.cuh>

// %RANGE% TUNE_THREADS_PER_BLOCK tpb 128:512:32
// %RANGE% TUNE_ITEMS_PER_THREAD ipt 7:24:1
// %RANGE% TUNE_RANK_ALGORITHM rank 0:1:1
// %RANGE% TUNE_LOG2_BATCH_TILES lbt 10:15:1

#if !TUNE_BASE
template <typename KeyT, typename OffsetT>
struct deferred_radix_policy_selector
{
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(cuda::compute_capability cc) const
    -> cub::RadixSortPolicy
  {
    auto policy = cub::detail::radix_sort::policy_selector_from_types<KeyT, cub::NullType, OffsetT, true>{}(cc);
    policy.onesweep.threads_per_block = TUNE_THREADS_PER_BLOCK;
    policy.onesweep.items_per_thread  = TUNE_ITEMS_PER_THREAD;
    policy.onesweep.tiles_per_batch   = 1 << TUNE_LOG2_BATCH_TILES;
    policy.onesweep.rank_algorithm =
      TUNE_RANK_ALGORITHM == 0 ? cub::RADIX_RANK_MATCH_EARLY_COUNTS_ANY : cub::RADIX_RANK_MATCH_EARLY_COUNTS_ATOMIC_OR;
    return policy;
  }
};
#endif // !TUNE_BASE

template <typename KeyT>
void radix_sort_keys_deferred(nvbench::state& state, nvbench::type_list<KeyT>)
try
{
  const auto elements       = static_cast<nvbench::uint64_t>(state.get_int64("Elements{io}"));
  const bool deferred       = state.get_string("Count") == "Deferred";
  const bit_entropy entropy = str_to_entropy(state.get_string("Entropy"));
  thrust::device_vector<KeyT> first(elements, thrust::no_init), second(elements, thrust::no_init);
  thrust::device_vector<nvbench::uint64_t> device_count(1, elements);
  cub::DoubleBuffer<KeyT> keys(first.data().get(), second.data().get());
  const auto count = cuda::args::deferred{device_count.data().get()};

  state.add_element_count(elements);
  state.add_global_memory_reads<KeyT>(elements, "Size");
  state.add_global_memory_writes<KeyT>(elements);

  // Both APIs use identical unsorted input for every sample. NVBench's manual
  // timer flushes L2 at start(), after generation and before the timed sort.
  caching_allocator_t alloc;
  state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
    keys.selector = 0;
    // Bound the shared generator's temporary allocation for 2^32-key inputs.
    constexpr std::size_t chunk_size = 1 << 24;
    seed_t seed{};
    for (std::size_t offset = 0; offset < first.size(); offset += chunk_size, ++seed)
    {
      const auto size = (cuda::std::min) (chunk_size, first.size() - offset);
      ::detail::gen_device(
        seed,
        cuda::std::span<KeyT>{first.data().get() + offset, size},
        entropy,
        (cuda::std::numeric_limits<KeyT>::min)(),
        (cuda::std::numeric_limits<KeyT>::max)());
    }
    timer.start();
    if (deferred)
    {
      auto env = cub_bench_env(
        alloc,
        launch
#if !TUNE_BASE
        ,
        cuda::execution::tune(deferred_radix_policy_selector<KeyT, nvbench::uint64_t>{})
#endif // !TUNE_BASE
      );
      _CCCL_TRY_RUNTIME_API(cub::DeviceRadixSort::SortKeys, "SortKeys failed", keys, count, 0, sizeof(KeyT) * 8, env);
    }
    else
    {
      auto env = cub_bench_env(alloc, launch);
      _CCCL_TRY_RUNTIME_API(cub::DeviceRadixSort::SortKeys, "SortKeys failed", keys, elements, 0, sizeof(KeyT) * 8, env);
    }
    timer.stop();
  });
}
catch (const std::bad_alloc&)
{
  state.skip("Skipping: out of memory.");
}

#ifdef TUNE_Key
using deferred_radix_key_types = nvbench::type_list<TUNE_Key>;
#else // !defined(TUNE_Key)
using deferred_radix_key_types = nvbench::type_list<nvbench::int32_t, nvbench::int64_t>;
#endif // TUNE_Key
NVBENCH_BENCH_TYPES(radix_sort_keys_deferred, NVBENCH_TYPE_AXES(deferred_radix_key_types))
  .set_name("base")
  .set_type_axes_names({"Key{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(20, 32, 1))
  .add_string_axis("Count", {"Immediate", "Deferred"})
  .add_string_axis("Entropy", {"1.000", "0.544", "0.201"});
