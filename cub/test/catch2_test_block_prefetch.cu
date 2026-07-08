// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <cub/detail/prefetch.cuh>
#include <cub/iterator/cache_modified_input_iterator.cuh>

#include <c2h/catch2_test_helper.h>

// BlockPrefetch emits global-memory prefetch *hints*: they carry no functional result and cannot be observed from
// the device program. A runtime test can therefore only assert that (1) issuing the hints never faults for any
// level / element type / tile shape, (2) the hints never disturb the data they cover, and (3) `none` and
// non-contiguous iterators (e.g. CacheModifiedInputIterator) compile and run as a genuine no-op. Whether the
// intended SASS is actually emitted is verified out-of-band (cuobjdump), not here.

// Compile-time coverage of the trait that gates every hint.
static_assert(cub::detail::can_prefetch_from<int*>, "raw pointers are contiguous and must be prefetchable");
static_assert(cub::detail::can_prefetch_from<const double*>, "const raw pointers must be prefetchable");
static_assert(!cub::detail::can_prefetch_from<cub::CacheModifiedInputIterator<cub::CacheLoadModifier::LOAD_CS, int>>,
              "CacheModifiedInputIterator routes through an explicit cache path and must be rejected");

// Prefetch the tile, then faithfully copy it through, so the launch is observable and any corruption of the
// prefetched region is caught by the host-side comparison.
template <typename T, int ThreadsInBlock, cub::detail::LoadPrefetch Level, int Stride, typename InputIteratorT>
__global__ void block_prefetch_kernel(InputIteratorT input, T* output, int num_items)
{
  cub::detail::BlockPrefetch<T, ThreadsInBlock, Level, Stride>::Prefetch(input, num_items);

  for (int i = static_cast<int>(threadIdx.x); i < num_items; i += ThreadsInBlock)
  {
    output[i] = input[i];
  }
}

template <typename T, int ThreadsInBlock, cub::detail::LoadPrefetch Level, int Stride = 128, typename InputIteratorT>
void test_block_prefetch(const c2h::device_vector<T>& d_input, InputIteratorT input)
{
  const int num_items = static_cast<int>(d_input.size());
  c2h::device_vector<T> d_output(num_items, T{});

  block_prefetch_kernel<T, ThreadsInBlock, Level, Stride>
    <<<1, ThreadsInBlock>>>(input, thrust::raw_pointer_cast(d_output.data()), num_items);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  // A prefetch hint must never alter the data it covers.
  REQUIRE(d_input == d_output);
}

using types            = c2h::type_list<std::uint8_t, std::int32_t, std::int64_t>;
using threads_in_block = c2h::enum_type_list<int, 32, 128>;
using load_prefetch_levels =
  c2h::enum_type_list<cub::detail::LoadPrefetch,
                      cub::detail::LoadPrefetch::none,
                      cub::detail::LoadPrefetch::l2,
                      cub::detail::LoadPrefetch::l1,
                      cub::detail::LoadPrefetch::bulk_l2>;

template <class TestType>
struct params_t
{
  using type = typename c2h::get<0, TestType>;

  static constexpr int threads_in_block            = c2h::get<1, TestType>::value;
  static constexpr cub::detail::LoadPrefetch level = c2h::get<2, TestType>::value;
};

C2H_TEST("BlockPrefetch issues hints without disturbing the tile",
         "[prefetch][block]",
         types,
         threads_in_block,
         load_prefetch_levels)
{
  using params = params_t<TestType>;
  using type   = typename params::type;

  // Cover the empty tile, single item, sub-block and ragged tiles, and tiles spanning many cache lines.
  constexpr int max_items = 8 * params::threads_in_block;
  const int num_items     = GENERATE_COPY(0, 1, 7, params::threads_in_block + 3, take(5, random(1, max_items)));
  CAPTURE(num_items);

  c2h::device_vector<type> d_input(num_items);
  c2h::gen(C2H_SEED(2), d_input);

  test_block_prefetch<type, params::threads_in_block, params::level>(d_input, thrust::raw_pointer_cast(d_input.data()));
}

C2H_TEST("BlockPrefetch is a no-op for CacheModifiedInputIterator", "[prefetch][block]", load_prefetch_levels)
{
  using type                                = int;
  constexpr int threads_in_block            = 128;
  constexpr cub::detail::LoadPrefetch level = c2h::get<0, TestType>::value;

  const int num_items = GENERATE_COPY(7, 200, take(3, random(1, 1024)));
  CAPTURE(num_items);

  c2h::device_vector<type> d_input(num_items);
  c2h::gen(C2H_SEED(2), d_input);

  // can_prefetch_from<CMI> is false, so Prefetch must compile out to nothing; the copy still reads through the CMI.
  cub::CacheModifiedInputIterator<cub::CacheLoadModifier::LOAD_CS, type> in(thrust::raw_pointer_cast(d_input.data()));
  test_block_prefetch<type, threads_in_block, level>(d_input, in);
}

C2H_TEST("BlockPrefetch handles unaligned tile bases", "[prefetch][block]", c2h::type_list<std::uint8_t, std::int32_t>)
{
  using type                     = c2h::get<0, TestType>;
  constexpr int threads_in_block = 128;

  // bulk_l2 aligns the base down to 16 B and extends the size to compensate; walk the base across a 16 B window.
  const int offset    = GENERATE(0, 1, 2, 3, 4, 5, 7, 8, 15);
  const int num_items = GENERATE(1, 33, 512);
  CAPTURE(offset, num_items);

  c2h::device_vector<type> d_storage(num_items + offset);
  c2h::gen(C2H_SEED(1), d_storage);

  // Prefetch/copy the sub-range that starts at an unaligned offset into the allocation.
  auto* base = thrust::raw_pointer_cast(d_storage.data()) + offset;
  c2h::device_vector<type> d_input(d_storage.begin() + offset, d_storage.end());

  test_block_prefetch<type, threads_in_block, cub::detail::LoadPrefetch::bulk_l2>(d_input, base);
}

C2H_TEST("BlockPrefetch honors a non-default stride", "[prefetch][block]")
{
  using type                     = int;
  constexpr int threads_in_block = 64;

  const int num_items = GENERATE_COPY(values({1, 100, 777}));
  CAPTURE(num_items);

  c2h::device_vector<type> d_input(num_items);
  c2h::gen(C2H_SEED(1), d_input);

  // A coarser 256 B stride issues fewer hints; the data must still be left intact.
  test_block_prefetch<type, threads_in_block, cub::detail::LoadPrefetch::l2, 256>(
    d_input, thrust::raw_pointer_cast(d_input.data()));
}
