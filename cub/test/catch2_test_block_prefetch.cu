// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/detail/prefetch.cuh>
#include <cub/iterator/cache_modified_input_iterator.cuh>

#include <thrust/device_vector.h>
#include <thrust/universal_vector.h>

#include <cuda/buffer>
#include <cuda/memory_resource>
#include <cuda/std/memory>
#include <cuda/stream>

#include <c2h/catch2_test_helper.h>

// BlockPrefetch emits global-memory prefetch *hints*: they carry no functional result and cannot be observed from
// the device program. A runtime test can therefore only assert that issuing the hints compiles and never faults for
// any level / element type / tile shape / iterator category, and that `none` and iterators rejected by
// `can_prefetch_from` (e.g. CacheModifiedInputIterator) compile as a genuine no-op. Each kernel copies its tile
// through so the launch has an observable, checkable result. Whether the intended SASS is actually emitted is
// verified out-of-band (cuobjdump), not here.

// Compile-time coverage of the trait that gates every hint.
static_assert(cub::detail::can_prefetch_from<int*>, "raw pointers are contiguous and must be prefetchable");
static_assert(cub::detail::can_prefetch_from<const double*>, "const raw pointers must be prefetchable");
static_assert(cub::detail::can_prefetch_from<thrust::device_vector<int>::iterator>,
              "thrust vector iterators are contiguous and must be prefetchable");
static_assert(cub::detail::can_prefetch_from<thrust::universal_vector<int>::iterator>,
              "thrust universal_vector iterators are contiguous and must be prefetchable");
static_assert(!cub::detail::can_prefetch_from<cub::CacheModifiedInputIterator<cub::CacheLoadModifier::LOAD_CS, int>>,
              "CacheModifiedInputIterator routes through an explicit cache path and must be rejected");

// Prefetch the tile, then copy it through so the launch has an observable result.
template <typename T, int ThreadsInBlock, cub::detail::LoadPrefetch Level, int Stride, typename InputIteratorT>
__global__ void block_prefetch_kernel(InputIteratorT input, T* output, int num_items)
{
  cub::detail::BlockPrefetch<ThreadsInBlock, Level, Stride>::Prefetch(input, num_items);

  for (int i = static_cast<int>(threadIdx.x); i < num_items; i += ThreadsInBlock)
  {
    output[i] = input[i];
  }
}

template <typename T, int ThreadsInBlock, cub::detail::LoadPrefetch Level, int Stride = 128, typename InputIteratorT>
void test_block_prefetch(const c2h::device_vector<T>& d_input, InputIteratorT input)
{
  const int num_items = static_cast<int>(d_input.size());
  c2h::device_vector<T> d_output(num_items, thrust::no_init);

  block_prefetch_kernel<T, ThreadsInBlock, Level, Stride>
    <<<1, ThreadsInBlock>>>(input, thrust::raw_pointer_cast(d_output.data()), num_items);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  REQUIRE(d_input == d_output);
}

using types            = c2h::type_list<uint8_t, int32_t, int64_t>;
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

C2H_TEST("BlockPrefetch runs for every level and tile shape",
         "[prefetch][block]",
         types,
         threads_in_block,
         load_prefetch_levels)
{
  using params = params_t<TestType>;
  using type   = typename params::type;

  // Each shape exercises a distinct address-math path: 0 = empty tile (size-0 TMA range), 1 = sub-cache-line,
  // 7 = sub-warp, threads + 3 = ragged tile, 8 * threads = multiple strided-loop iterations per thread.
  const int num_items = GENERATE_COPY(0, 1, 7, params::threads_in_block + 3, 8 * params::threads_in_block);
  CAPTURE(num_items);

  c2h::device_vector<type> d_input(num_items, thrust::no_init);
  c2h::gen(C2H_SEED(2), d_input);

  test_block_prefetch<type, params::threads_in_block, params::level>(d_input, thrust::raw_pointer_cast(d_input.data()));
}

C2H_TEST("BlockPrefetch is a no-op for CacheModifiedInputIterator", "[prefetch][block]", load_prefetch_levels)
{
  using type                                = int;
  constexpr int threads_in_block            = 128;
  constexpr cub::detail::LoadPrefetch level = c2h::get<0, TestType>::value;

  // The no-op path has no size-dependent behavior; one size suffices.
  const int num_items = 200;
  CAPTURE(num_items, threads_in_block, level);

  c2h::device_vector<type> d_input(num_items, thrust::no_init);
  c2h::gen(C2H_SEED(2), d_input);

  // can_prefetch_from<CMI> is false, so Prefetch must compile out to nothing; the copy still reads through the CMI.
  cub::CacheModifiedInputIterator<cub::CacheLoadModifier::LOAD_CS, type> in(thrust::raw_pointer_cast(d_input.data()));
  test_block_prefetch<type, threads_in_block, level>(d_input, in);
}

C2H_TEST("BlockPrefetch works with thrust vector iterators", "[prefetch][block]", load_prefetch_levels)
{
  using type                                = int;
  constexpr int threads_in_block            = 128;
  constexpr cub::detail::LoadPrefetch level = c2h::get<0, TestType>::value;

  // A sub-line tile and a multi-line tile cover both strided-loop shapes through wrapped iterators.
  const int num_items = GENERATE(7, 200);
  CAPTURE(num_items, threads_in_block, level);

  c2h::device_vector<type> d_input(num_items, thrust::no_init);
  c2h::gen(C2H_SEED(2), d_input);

  // thrust vector iterators are contiguous (thrust::is_contiguous_iterator), so hints flow through them.
  STATIC_REQUIRE(cub::detail::can_prefetch_from<decltype(d_input.cbegin())>);
  test_block_prefetch<type, threads_in_block, level>(d_input, d_input.cbegin());

  thrust::universal_vector<type> universal_input(d_input.begin(), d_input.end());
  STATIC_REQUIRE(cub::detail::can_prefetch_from<decltype(universal_input.cbegin())>);
  test_block_prefetch<type, threads_in_block, level>(d_input, universal_input.cbegin());
}

C2H_TEST("BlockPrefetch works with cuda::buffer iterators", "[prefetch][block]", load_prefetch_levels)
{
  using type                                = int;
  constexpr int threads_in_block            = 128;
  constexpr cub::detail::LoadPrefetch level = c2h::get<0, TestType>::value;

  // A sub-line tile and a multi-line tile cover both strided-loop shapes through the buffer iterator.
  const int num_items = GENERATE(7, 200);
  CAPTURE(num_items, threads_in_block, level);

  c2h::device_vector<type> d_input(num_items, thrust::no_init);
  c2h::gen(C2H_SEED(2), d_input);

  // Managed memory is host- and device-accessible, so the buffer can be filled from the host and read in the kernel.
  cuda::mr::legacy_managed_memory_resource mr;
  auto buf =
    cuda::make_buffer<type>(cuda::stream_ref{cudaStream_t{}}, mr, static_cast<size_t>(num_items), cuda::no_init);
  REQUIRE(cudaSuccess
          == cudaMemcpy(cuda::std::to_address(buf.begin()),
                        thrust::raw_pointer_cast(d_input.data()),
                        num_items * sizeof(type),
                        cudaMemcpyDefault));

  // cuda::buffer's heterogeneous_iterator is a contiguous iterator, so the trait accepts it and hints are emitted.
  STATIC_REQUIRE(cub::detail::can_prefetch_from<decltype(buf.begin())>);
  test_block_prefetch<type, threads_in_block, level>(d_input, buf.begin());
}

C2H_TEST("BlockPrefetch handles unaligned tile bases", "[prefetch][block]", c2h::type_list<uint8_t, int32_t>)
{
  using type                     = c2h::get<0, TestType>;
  constexpr int threads_in_block = 128;

  // bulk_l2 aligns the base down to 16 B and extends the size to compensate. The offsets cover the
  // equivalence classes of the 16 B window: aligned, minimal, mid, and maximal misalignment.
  const int offset    = GENERATE(0, 1, 8, 15);
  const int num_items = GENERATE(1, 33, 512);
  CAPTURE(offset, num_items, threads_in_block);

  c2h::device_vector<type> d_storage(num_items + offset, thrust::no_init);
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

  const int num_items = GENERATE(1, 100, 777);
  CAPTURE(num_items, threads_in_block);

  c2h::device_vector<type> d_input(num_items, thrust::no_init);
  c2h::gen(C2H_SEED(1), d_input);

  // A coarser 256 B stride issues fewer hints; the copy-through must still be correct.
  test_block_prefetch<type, threads_in_block, cub::detail::LoadPrefetch::l2, 256>(
    d_input, thrust::raw_pointer_cast(d_input.data()));
}
