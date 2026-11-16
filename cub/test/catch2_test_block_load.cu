// SPDX-FileCopyrightText: Copyright (c) 2011-2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <cub/block/block_load.cuh>
#include <cub/iterator/cache_modified_input_iterator.cuh>
#include <cub/util_allocator.cuh>
#include <cub/util_arch.cuh>

#include <cuda/iterator>
#include <cuda/std/span>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/type_traits>

#include "cub_test_macros.h"
#include <c2h/extended_types.h>

template <typename T, int ItemsPerThread, int ThreadsInBlock, cub::BlockLoadAlgorithm LoadAlgorithm>
static __device__ int get_output_idx(int item, bool full_tile)
{
  if constexpr (LoadAlgorithm == cub::BlockLoadAlgorithm::BLOCK_LOAD_STRIPED)
  {
    return static_cast<int>(threadIdx.x) + (ThreadsInBlock * item);
  }
  else if constexpr (LoadAlgorithm == cub::BlockLoadAlgorithm::BLOCK_LOAD_STRIPED_VECTORIZED)
  {
    // Full tiles are striped vectors of up to four device words of T, or of single items if they do not hold a whole
    // number of T. Guarded loads are striped.
    using vector_t             = cub::detail::block_load_vector_t<T, ItemsPerThread>;
    constexpr int vec_size     = sizeof(vector_t) % sizeof(T) == 0 ? static_cast<int>(sizeof(vector_t) / sizeof(T)) : 1;
    const int items_per_vector = full_tile ? vec_size : 1;
    const int vector           = item / items_per_vector;
    const int element          = item % items_per_vector;
    return ((static_cast<int>(threadIdx.x) + (ThreadsInBlock * vector)) * items_per_vector) + element;
  }
  else
  {
    return (static_cast<int>(threadIdx.x) * ItemsPerThread) + item;
  }
}

template <int ItemsPerThread,
          int ThreadsInBlock,
          cub::BlockLoadAlgorithm LoadAlgorithm,
          typename InputIteratorT,
          typename OutputIteratorT>
__global__ void kernel(cuda::std::true_type, InputIteratorT input, OutputIteratorT output, int num_items)
{
  using input_t      = cub::detail::it_value_t<InputIteratorT>;
  using block_load_t = cub::BlockLoad<input_t, ThreadsInBlock, ItemsPerThread, LoadAlgorithm>;
  using storage_t    = typename block_load_t::TempStorage;

  __shared__ storage_t storage;
  block_load_t block_load(storage);
  input_t data[ItemsPerThread];

  const bool full_tile = ItemsPerThread * ThreadsInBlock == num_items;
  if (full_tile)
  {
    block_load.Load(input, data);
  }
  else
  {
    block_load.Load(input, data, num_items);
  }

  for (int i = 0; i < ItemsPerThread; i++)
  {
    const int idx = get_output_idx<input_t, ItemsPerThread, ThreadsInBlock, LoadAlgorithm>(i, full_tile);
    if (idx < num_items)
    {
      output[idx] = data[i];
    }
  }
}

template <int ItemsPerThread,
          int ThreadsInBlock,
          cub::BlockLoadAlgorithm /* LoadAlgorithm */,
          typename InputIteratorT,
          typename OutputIteratorT>
__global__ void kernel(cuda::std::false_type, InputIteratorT input, OutputIteratorT output, int num_items)
{
  using input_t = cub::detail::it_value_t<InputIteratorT>;
  for (int i = 0; i < ItemsPerThread; i++)
  {
    const int idx =
      get_output_idx<input_t, ItemsPerThread, ThreadsInBlock, cub::BlockLoadAlgorithm::BLOCK_LOAD_DIRECT>(i, true);
    if (idx < num_items)
    {
      output[idx] = input[idx];
    }
  }
}

template <int ItemsPerThread, int ThreadsInBlock, cub::BlockLoadAlgorithm LoadAlgorithm, typename T, typename InputIteratorT>
void test_block_load(const c2h::device_vector<T>& d_input, InputIteratorT input)
{
  using block_load_t = cub::BlockLoad<T, ThreadsInBlock, ItemsPerThread, LoadAlgorithm>;
  using storage_t    = typename block_load_t::TempStorage;
  constexpr auto sufficient_resources =
    cuda::std::bool_constant<sizeof(storage_t) <= cub::detail::max_smem_per_block>{};

  c2h::device_vector<T> d_output(d_input.size());
  kernel<ItemsPerThread, ThreadsInBlock, LoadAlgorithm><<<1, ThreadsInBlock>>>(
    sufficient_resources, input, thrust::raw_pointer_cast(d_output.data()), static_cast<int>(d_input.size()));
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
  REQUIRE(d_input == d_output);
}

// %PARAM% IPT it 1:11

using types     = c2h::type_list<std::uint8_t, std::int32_t, std::int64_t>;
using vec_types = c2h::type_list<long2, double2>;

using even_threads_in_block = c2h::enum_type_list<int, 32, 128>;
using odd_threads_in_block  = c2h::enum_type_list<int, 15, 65>;
using a_block_size          = c2h::enum_type_list<int, 256>;

using items_per_thread = c2h::enum_type_list<int, IPT>;
using load_algorithm =
  c2h::enum_type_list<cub::BlockLoadAlgorithm,
                      cub::BlockLoadAlgorithm::BLOCK_LOAD_DIRECT,
                      cub::BlockLoadAlgorithm::BLOCK_LOAD_STRIPED,
                      cub::BlockLoadAlgorithm::BLOCK_LOAD_VECTORIZE,
                      cub::BlockLoadAlgorithm::BLOCK_LOAD_STRIPED_VECTORIZED,
                      cub::BlockLoadAlgorithm::BLOCK_LOAD_TRANSPOSE,
                      cub::BlockLoadAlgorithm::BLOCK_LOAD_WARP_TRANSPOSE,
                      cub::BlockLoadAlgorithm::BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED>;

using odd_load_algorithm =
  c2h::enum_type_list<cub::BlockLoadAlgorithm,
                      cub::BlockLoadAlgorithm::BLOCK_LOAD_DIRECT,
                      cub::BlockLoadAlgorithm::BLOCK_LOAD_STRIPED,
                      cub::BlockLoadAlgorithm::BLOCK_LOAD_VECTORIZE,
                      cub::BlockLoadAlgorithm::BLOCK_LOAD_STRIPED_VECTORIZED,
                      cub::BlockLoadAlgorithm::BLOCK_LOAD_TRANSPOSE>;

using vectorized_load_algorithm =
  c2h::enum_type_list<cub::BlockLoadAlgorithm,
                      cub::BlockLoadAlgorithm::BLOCK_LOAD_VECTORIZE,
                      cub::BlockLoadAlgorithm::BLOCK_LOAD_STRIPED_VECTORIZED>;

template <class TestType>
struct params_t
{
  using type = typename c2h::get<0, TestType>;

  static constexpr int items_per_thread                   = c2h::get<1, TestType>::value;
  static constexpr int threads_in_block                   = c2h::get<2, TestType>::value;
  static constexpr int tile_size                          = items_per_thread * threads_in_block;
  static constexpr cub::BlockLoadAlgorithm load_algorithm = c2h::get<3, TestType>::value;
};

CUB_TEST("Block load works with even block sizes",
         "[load][block]",
         CUB_SMALL,
         types,
         items_per_thread,
         even_threads_in_block,
         load_algorithm)
{
  using params = params_t<TestType>;
  using type   = typename params::type;

  c2h::device_vector<type> d_input(GENERATE_COPY(take(10, random(0, params::tile_size))));
  c2h::gen(C2H_SEED(10), d_input);

  test_block_load<params::items_per_thread, params::threads_in_block, params::load_algorithm>(
    d_input, thrust::raw_pointer_cast(d_input.data()));
}

CUB_TEST("Block load works with even odd sizes",
         "[load][block]",
         CUB_SMALL,
         types,
         items_per_thread,
         odd_threads_in_block,
         odd_load_algorithm)
{
  using params = params_t<TestType>;
  using type   = typename params::type;

  c2h::device_vector<type> d_input(GENERATE_COPY(take(10, random(0, params::tile_size))));
  c2h::gen(C2H_SEED(10), d_input);
  test_block_load<params::items_per_thread, params::threads_in_block, params::load_algorithm>(
    d_input, thrust::raw_pointer_cast(d_input.data()));
}

// WAR bug in vec type handling in NVCC 12.0 + GCC 11.4 + C++20
#if !(_CCCL_CUDA_COMPILER(NVCC, ==, 12, 0) && _CCCL_COMPILER(GCC, ==, 11, 4) && _CCCL_STD_VER == 2020)
CUB_TEST("Block load works with even vector types",
         "[load][block]",
         CUB_SMALL,
         vec_types,
         items_per_thread,
         a_block_size,
         load_algorithm)
{
  using params = params_t<TestType>;
  using type   = typename params::type;

  c2h::device_vector<type> d_input(GENERATE_COPY(take(10, random(0, params::tile_size))));
  c2h::gen(C2H_SEED(10), d_input);
  test_block_load<params::items_per_thread, params::threads_in_block, params::load_algorithm>(
    d_input, thrust::raw_pointer_cast(d_input.data()));
}
#endif // !(NVCC 12.0 and GCC 11.4 and C++20)

CUB_TEST("Block load works with custom types", "[load][block]", CUB_SMALL, items_per_thread, load_algorithm)
{
  using type                                              = c2h::custom_type_t<c2h::equal_comparable_t>;
  constexpr int items_per_thread                          = c2h::get<0, TestType>::value;
  constexpr int threads_in_block                          = 64;
  constexpr int tile_size                                 = items_per_thread * threads_in_block;
  static constexpr cub::BlockLoadAlgorithm load_algorithm = c2h::get<1, TestType>::value;

  c2h::device_vector<type> d_input(GENERATE_COPY(take(10, random(0, tile_size))));
  c2h::gen(C2H_SEED(10), d_input);
  test_block_load<items_per_thread, threads_in_block, load_algorithm>(d_input, thrust::raw_pointer_cast(d_input.data()));
}

CUB_TEST("Block load works with caching iterators", "[load][block]", CUB_SMALL, items_per_thread, load_algorithm)
{
  using type                                              = int;
  constexpr int items_per_thread                          = c2h::get<0, TestType>::value;
  constexpr int threads_in_block                          = 64;
  constexpr int tile_size                                 = items_per_thread * threads_in_block;
  static constexpr cub::BlockLoadAlgorithm load_algorithm = c2h::get<1, TestType>::value;

  c2h::device_vector<type> d_input(GENERATE_COPY(take(10, random(0, tile_size))));
  c2h::gen(C2H_SEED(10), d_input);
  const cub::CacheModifiedInputIterator<cub::CacheLoadModifier::LOAD_DEFAULT, type> in(
    thrust::raw_pointer_cast(d_input.data()));
  test_block_load<items_per_thread, threads_in_block, load_algorithm>(d_input, in);
}

CUB_TEST("Block load works with fancy iterators", "[load][block]", CUB_SMALL, items_per_thread, load_algorithm)
{
  using type                                              = int;
  constexpr int items_per_thread                          = c2h::get<0, TestType>::value;
  constexpr int threads_in_block                          = 64;
  constexpr int tile_size                                 = items_per_thread * threads_in_block;
  static constexpr cub::BlockLoadAlgorithm load_algorithm = c2h::get<1, TestType>::value;

  c2h::device_vector<type> d_input(GENERATE_COPY(take(10, random(0, tile_size))));
  c2h::gen(c2h::modulo_t{d_input.size()}, d_input);
  test_block_load<items_per_thread, threads_in_block, load_algorithm>(d_input, cuda::counting_iterator<type>{0});
}

#if IPT == 1
CUB_TEST("Vectorized block load with const, non-const and cache modified inputs and different alignment cases",
         "[load][block]",
         CUB_SMALL,
         c2h::type_list<const int*, int*, cub::CacheModifiedInputIterator<cub::CacheLoadModifier::LOAD_LDG, int>>,
         vectorized_load_algorithm,
         c2h::enum_type_list<int, 4, 8>)
{
  using type           = int;
  using input_ptr_type = c2h::get<0, TestType>;

  const int offset_for_elements                           = GENERATE_COPY(0, 1, 2, 3, 4);
  constexpr int items_per_thread                          = c2h::get<2, TestType>::value;
  constexpr int threads_in_block                          = 64;
  constexpr int tile_size                                 = items_per_thread * threads_in_block;
  static constexpr cub::BlockLoadAlgorithm load_algorithm = c2h::get<1, TestType>::value;

  c2h::device_vector<type> d_input_ref(tile_size);
  c2h::gen(C2H_SEED(10), d_input_ref);

  c2h::device_vector<type> d_input(tile_size + offset_for_elements);
  thrust::copy_n(d_input_ref.begin(), tile_size, d_input.begin() + offset_for_elements);

  test_block_load<items_per_thread, threads_in_block, load_algorithm, type, input_ptr_type>(
    d_input_ref, thrust::raw_pointer_cast(d_input.data()) + offset_for_elements);
}

template <typename T, int ItemsPerThread, int ThreadsInBlock, cub::BlockLoadAlgorithm LoadAlgorithm, typename InputIteratorT>
__global__ void vectorized_full_tile_load_kernel(InputIteratorT input, T* output)
{
  T data[ItemsPerThread];
  cub::BlockLoad<T, ThreadsInBlock, ItemsPerThread, LoadAlgorithm>().Load(input, data);
  for (int i = 0; i < ItemsPerThread; i++)
  {
    output[get_output_idx<T, ItemsPerThread, ThreadsInBlock, LoadAlgorithm>(i, true)] = data[i];
  }
}

template <typename T, cub::BlockLoadAlgorithm LoadAlgorithm, typename InputT, typename InputIteratorT>
void test_vectorized_full_tile_load(const c2h::device_vector<InputT>& d_input, InputIteratorT input)
{
  constexpr int items_per_thread = 4;
  constexpr int threads_in_block = 64;
  REQUIRE(d_input.size() == items_per_thread * threads_in_block);

  c2h::device_vector<T> d_output(d_input.size(), thrust::no_init);
  vectorized_full_tile_load_kernel<T, items_per_thread, threads_in_block, LoadAlgorithm>
    <<<1, threads_in_block>>>(input, thrust::raw_pointer_cast(d_output.data()));
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
  const c2h::device_vector<T> expected(d_input.begin(), d_input.end());
  REQUIRE(expected == d_output);
}

CUB_TEST("Vectorized block load converts from iterators of a different value type",
         "[load][block]",
         CUB_SMALL,
         vectorized_load_algorithm)
{
  static constexpr cub::BlockLoadAlgorithm load_algorithm = c2h::get<0, TestType>::value;

  c2h::device_vector<int> d_input(256, thrust::no_init);
  c2h::gen(C2H_SEED(1), d_input);
  int* ptr = thrust::raw_pointer_cast(d_input.data());

  test_vectorized_full_tile_load<long long, load_algorithm>(d_input, ptr);
  test_vectorized_full_tile_load<long long, load_algorithm>(d_input, cuda::std::span<int>(ptr, d_input.size()).begin());
  test_vectorized_full_tile_load<long long, load_algorithm>(
    d_input, cub::CacheModifiedInputIterator<cub::CacheLoadModifier::LOAD_DEFAULT, int>(ptr));
  test_vectorized_full_tile_load<int, load_algorithm>(
    d_input, cuda::std::span<volatile int>(ptr, d_input.size()).begin());
  test_vectorized_full_tile_load<int, load_algorithm>(d_input, static_cast<volatile int*>(ptr));
}

template <int VecSize, int ItemsPerThread, cub::CacheLoadModifier Modifier, typename T, typename InputIteratorT>
__global__ void load_direct_blocked_vectorized_kernel(InputIteratorT input, T* output)
{
  T data[ItemsPerThread];
  cub::LoadDirectBlockedVectorized<Modifier, VecSize>(static_cast<int>(threadIdx.x), input, data);
  for (int i = 0; i < ItemsPerThread; i++)
  {
    output[(threadIdx.x * ItemsPerThread) + i] = data[i];
  }
}

template <int VecSize,
          int ItemsPerThread,
          int ThreadsInBlock,
          cub::CacheLoadModifier Modifier = cub::LOAD_DEFAULT,
          typename T,
          typename InputIteratorT>
void test_load_direct_blocked_vectorized(const c2h::device_vector<T>& d_expected, InputIteratorT input)
{
  REQUIRE(d_expected.size() == ItemsPerThread * ThreadsInBlock);

  c2h::device_vector<T> d_output(d_expected.size(), thrust::no_init);
  load_direct_blocked_vectorized_kernel<VecSize, ItemsPerThread, Modifier>
    <<<1, ThreadsInBlock>>>(input, thrust::raw_pointer_cast(d_output.data()));
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
  REQUIRE(d_expected == d_output);
}

template <int ItemsPerThread, typename T>
__global__ void load_direct_blocked_vectorized_type_first_kernel(T* input, T* output)
{
  T data[ItemsPerThread];
  cub::LoadDirectBlockedVectorized<T, ItemsPerThread>(static_cast<int>(threadIdx.x), input, data);
  for (int i = 0; i < ItemsPerThread; i++)
  {
    output[(threadIdx.x * ItemsPerThread) + i] = data[i];
  }
}

CUB_TEST("LoadDirectBlockedVectorized accepts an explicitly specified data type and items per thread",
         "[load][block]",
         CUB_SMALL,
         types)
{
  using type                     = c2h::get<0, TestType>;
  constexpr int items_per_thread = 4;
  constexpr int threads_in_block = 64;

  c2h::device_vector<type> d_input(items_per_thread * threads_in_block, thrust::no_init);
  c2h::gen(C2H_SEED(1), d_input);
  c2h::device_vector<type> d_output(d_input.size(), thrust::no_init);
  load_direct_blocked_vectorized_type_first_kernel<items_per_thread>
    <<<1, threads_in_block>>>(thrust::raw_pointer_cast(d_input.data()), thrust::raw_pointer_cast(d_output.data()));
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
  REQUIRE(d_input == d_output);
}

// Offset 0 takes the vectorized loads. Offset 1 misaligns the vectors of multiple items, which use the scalar fallback.
CUB_TEST("LoadDirectBlockedVectorized loads the items of a thread with vectors of VecSize items",
         "[load][block]",
         CUB_SMALL,
         types)
{
  using type                     = c2h::get<0, TestType>;
  constexpr int items_per_thread = 16;
  constexpr int threads_in_block = 64;
  constexpr int tile_size        = items_per_thread * threads_in_block;

  const int offset = GENERATE(0, 1);
  c2h::device_vector<type> d_expected(tile_size, thrust::no_init);
  c2h::gen(C2H_SEED(1), d_expected);
  c2h::device_vector<type> d_input(tile_size + offset, thrust::no_init);
  thrust::copy_n(d_expected.begin(), tile_size, d_input.begin() + offset);
  const type* input = thrust::raw_pointer_cast(d_input.data()) + offset;

  test_load_direct_blocked_vectorized<0, items_per_thread, threads_in_block>(d_expected, input);
  test_load_direct_blocked_vectorized<1, items_per_thread, threads_in_block>(d_expected, input);
  test_load_direct_blocked_vectorized<4, items_per_thread, threads_in_block>(d_expected, input);
  test_load_direct_blocked_vectorized<16, items_per_thread, threads_in_block>(d_expected, input);
}

template <typename InputT, typename T>
struct conversion
{
  using input_t  = InputT;
  using output_t = T;
};

using conversions =
  c2h::type_list<conversion<std::int32_t, std::int64_t>,
                 conversion<std::int64_t, std::int16_t>,
                 conversion<std::int16_t, std::int64_t>,
                 conversion<std::int32_t, std::uint32_t>,
                 conversion<std::uint8_t, std::int32_t>,
                 conversion<std::int16_t, std::uint64_t>,
                 conversion<std::uint64_t, std::int8_t>>;

// Offset 0 takes the vectorized loads. Offset 1 misaligns the vectors of multiple items, which use the scalar fallback.
CUB_TEST(
  "LoadDirectBlockedVectorized converts the items to the destination type", "[load][block]", CUB_SMALL, conversions)
{
  using input_t                  = typename c2h::get<0, TestType>::input_t;
  using output_t                 = typename c2h::get<0, TestType>::output_t;
  constexpr int items_per_thread = 4;
  constexpr int threads_in_block = 64;
  constexpr int tile_size        = items_per_thread * threads_in_block;

  const int offset = GENERATE(0, 1);
  c2h::device_vector<input_t> d_input(tile_size + offset, thrust::no_init);
  c2h::gen(C2H_SEED(1), d_input);
  const c2h::device_vector<output_t> d_expected(d_input.begin() + offset, d_input.end());
  const input_t* input = thrust::raw_pointer_cast(d_input.data()) + offset;

  test_load_direct_blocked_vectorized<0, items_per_thread, threads_in_block>(d_expected, input);
  test_load_direct_blocked_vectorized<0, items_per_thread, threads_in_block, cub::LOAD_LDG>(d_expected, input);
}

// Places the items of a thread after a byte, so that they are only aligned to their type
template <typename T, int ItemsPerThread>
struct items_after_a_byte
{
  char byte;
  T items[ItemsPerThread];
};

template <cub::CacheLoadModifier Modifier, int ItemsPerThread, int ThreadsInBlock, typename T>
__global__ void load_direct_blocked_vectorized_unaligned_destination_kernel(const T* input, T* output)
{
  __shared__ items_after_a_byte<T, ItemsPerThread> shared[ThreadsInBlock];
  cub::LoadDirectBlockedVectorized<Modifier>(static_cast<int>(threadIdx.x), input, shared[threadIdx.x].items);
  for (int i = 0; i < ItemsPerThread; i++)
  {
    output[(threadIdx.x * ItemsPerThread) + i] = shared[threadIdx.x].items[i];
  }
}

// A class type without padding, which has unique object representations and is aligned to 2 bytes
struct int16_pair
{
  std::int16_t first;
  std::int16_t second;

  friend __host__ __device__ bool operator==(int16_pair lhs, int16_pair rhs)
  {
    return lhs.first == rhs.first && lhs.second == rhs.second;
  }
};
static_assert(cuda::std::has_unique_object_representations_v<int16_pair>);

using unaligned_destination_types =
  c2h::type_list<std::int8_t,
                 std::int16_t,
                 char2,
                 char3,
                 short3,
                 int16_pair
#  if TEST_HALF_T()
                 ,
                 half_t
#  endif // TEST_HALF_T()
                 >;

struct non_default_constructible
{
  int value;

  __host__ __device__ explicit non_default_constructible(int value)
      : value(value)
  {}

  friend __host__ __device__ bool operator==(non_default_constructible lhs, non_default_constructible rhs)
  {
    return lhs.value == rhs.value;
  }
};
static_assert(!cuda::std::is_default_constructible_v<non_default_constructible>);
static_assert(cuda::is_trivially_copyable_v<non_default_constructible>); // CUB uses this check to vectorize

template <cub::CacheLoadModifier Modifier, int VecSize, typename T, int... Is>
__global__ void load_direct_blocked_vectorized_non_default_constructible_kernel(
  const T* input, T* output, cuda::std::integer_sequence<int, Is...>)
{
  constexpr int items_per_thread = sizeof...(Is);
  // T has no default constructor, thus every item needs an initializer
  T data[items_per_thread] = {T{Is}...};
  cub::LoadDirectBlockedVectorized<Modifier, VecSize>(static_cast<int>(threadIdx.x), input, data);
  for (int i = 0; i < items_per_thread; i++)
  {
    output[(threadIdx.x * items_per_thread) + i] = data[i];
  }
}

template <cub::CacheLoadModifier Modifier, int VecSize, int ItemsPerThread, int ThreadsInBlock, typename T>
void test_load_direct_blocked_vectorized_non_default_constructible(const c2h::host_vector<T>& h_expected, const T* input)
{
  REQUIRE(h_expected.size() == ItemsPerThread * ThreadsInBlock);

  c2h::device_vector<T> d_output(h_expected.size(), T{-1});
  load_direct_blocked_vectorized_non_default_constructible_kernel<Modifier, VecSize><<<1, ThreadsInBlock>>>(
    input, thrust::raw_pointer_cast(d_output.data()), cuda::std::make_integer_sequence<int, ItemsPerThread>{});
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
  // Compares on the host, because Catch2 cannot print the device references of a type without a stream insertion
  REQUIRE(c2h::host_vector<T>(d_output) == h_expected);
}

using load_modifiers =
  c2h::enum_type_list<cub::CacheLoadModifier,
                      cub::LOAD_DEFAULT,
                      cub::LOAD_CA,
                      cub::LOAD_CG,
                      cub::LOAD_CS,
                      cub::LOAD_CV,
                      cub::LOAD_LDG,
                      cub::LOAD_VOLATILE>;

template <int ThreadsInBlock, int VecSize, cub::CacheLoadModifier Modifier, int ItemsPerThread, typename InputT, typename T>
__global__ void load_direct_striped_vectorized_kernel(const InputT* input, T* output)
{
  T data[ItemsPerThread];
  cub::LoadDirectStripedVectorized<ThreadsInBlock, VecSize, Modifier>(static_cast<int>(threadIdx.x), input, data);
  for (int i = 0; i < ItemsPerThread; i++)
  {
    output[((threadIdx.x + (ThreadsInBlock * (i / VecSize))) * VecSize) + (i % VecSize)] = data[i];
  }
}

template <int VecSize, cub::CacheLoadModifier Modifier = cub::CacheLoadModifier::LOAD_DEFAULT, typename T, typename InputT>
void test_load_direct_striped_vectorized(const c2h::device_vector<T>& d_expected, const InputT* input)
{
  constexpr int items_per_thread = 8;
  constexpr int threads_in_block = 64;
  REQUIRE(d_expected.size() == items_per_thread * threads_in_block);

  c2h::device_vector<T> d_output(d_expected.size(), thrust::no_init);
  load_direct_striped_vectorized_kernel<threads_in_block, VecSize, Modifier, items_per_thread>
    <<<1, threads_in_block>>>(input, thrust::raw_pointer_cast(d_output.data()));
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
  REQUIRE(d_expected == d_output);
}

// Offset 0 takes the vectorized loads. Offset 1 misaligns the vectors of multiple items, which are loaded item by item.
CUB_TEST("LoadDirectStripedVectorized loads a striped arrangement of vectors of VecSize items",
         "[load][block]",
         CUB_SMALL,
         types)
{
  using type              = c2h::get<0, TestType>;
  constexpr int tile_size = 8 * 64;

  const int offset = GENERATE(0, 1);
  c2h::device_vector<type> d_expected(tile_size);
  c2h::gen(C2H_SEED(1), d_expected);
  c2h::device_vector<type> d_input(tile_size + offset);
  thrust::copy_n(d_expected.begin(), tile_size, d_input.begin() + offset);
  const type* input = thrust::raw_pointer_cast(d_input.data()) + offset;

  test_load_direct_striped_vectorized<1>(d_expected, input);
  test_load_direct_striped_vectorized<2>(d_expected, input);
  test_load_direct_striped_vectorized<4>(d_expected, input);
  test_load_direct_striped_vectorized<4, cub::CacheLoadModifier::LOAD_LDG>(d_expected, input);
}

CUB_TEST_CASE("LoadDirectStripedVectorized converts the items to the destination type", "[load][block]", CUB_SMALL)
{
  constexpr int tile_size = 8 * 64;

  const int offset = GENERATE(0, 1);
  c2h::device_vector<int> d_input(tile_size + offset);
  c2h::gen(C2H_SEED(1), d_input);
  const c2h::device_vector<long long> d_expected(d_input.begin() + offset, d_input.end());
  const int* input = thrust::raw_pointer_cast(d_input.data()) + offset;

  test_load_direct_striped_vectorized<4>(d_expected, input);
}

// Items aligned to 1 or 2 bytes are copied as wider words, but the destination items are only aligned to their type.
// Offset 0 takes the vectorized loads. Offset 1 misaligns the vectors of multiple items, which are loaded item by item.
CUB_TEST("LoadDirectBlockedVectorized loads into items aligned only to their type",
         "[load][block]",
         CUB_SMALL,
         unaligned_destination_types,
         c2h::enum_type_list<cub::CacheLoadModifier, cub::LOAD_DEFAULT, cub::LOAD_LDG>)
{
  using type                     = c2h::get<0, TestType>;
  constexpr auto modifier        = c2h::get<1, TestType>::value;
  constexpr int items_per_thread = 16;
  constexpr int threads_in_block = 64;
  constexpr int tile_size        = items_per_thread * threads_in_block;

  const int offset = GENERATE(0, 1);
  c2h::device_vector<type> d_expected(tile_size, thrust::no_init);
  if constexpr (cuda::std::is_same_v<type, int16_pair>)
  {
    // c2h generates no int16_pair, thus its members are generated as int16_t
    c2h::device_vector<std::int16_t> d_members(2 * tile_size, thrust::no_init);
    c2h::gen(C2H_SEED(1), d_members);
    REQUIRE(cudaSuccess
            == cudaMemcpy(thrust::raw_pointer_cast(d_expected.data()),
                          thrust::raw_pointer_cast(d_members.data()),
                          tile_size * sizeof(type),
                          cudaMemcpyDeviceToDevice));
  }
  else
  {
    c2h::gen(C2H_SEED(1), d_expected);
  }
  c2h::device_vector<type> d_input(tile_size + offset, thrust::no_init);
  thrust::copy_n(d_expected.begin(), tile_size, d_input.begin() + offset);
  const type* input = thrust::raw_pointer_cast(d_input.data()) + offset;

  c2h::device_vector<type> d_blocked_output(tile_size, thrust::no_init);
  load_direct_blocked_vectorized_unaligned_destination_kernel<modifier, items_per_thread, threads_in_block>
    <<<1, threads_in_block>>>(input, thrust::raw_pointer_cast(d_blocked_output.data()));
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
  // Compares on the host, because Catch2 cannot print the device references of a type without a stream insertion
  const c2h::host_vector<type> h_expected(d_expected);
  REQUIRE(c2h::host_vector<type>(d_blocked_output) == h_expected);
}

// Offset 0 takes the vectorized loads with vectors of 2, 4 and 8 items. Offset 1 misaligns the vectors of multiple
// items, which are loaded item by item.
CUB_TEST("LoadDirectBlockedVectorized loads non-default-constructible types with every cache load modifier",
         "[load][block]",
         CUB_SMALL,
         load_modifiers)
{
  using type                     = non_default_constructible;
  constexpr auto modifier        = c2h::get<0, TestType>::value;
  constexpr int items_per_thread = 8;
  constexpr int threads_in_block = 64;
  constexpr int tile_size        = items_per_thread * threads_in_block;

  const int offset = GENERATE(0, 1);
  c2h::host_vector<type> h_input;
  for (int i = 0; i < tile_size + offset; i++)
  {
    h_input.push_back(type{i});
  }
  const c2h::device_vector<type> d_input(h_input);
  const c2h::host_vector<type> h_expected(h_input.begin() + offset, h_input.end());
  const type* input = thrust::raw_pointer_cast(d_input.data()) + offset;

  test_load_direct_blocked_vectorized_non_default_constructible<modifier, 0, items_per_thread, threads_in_block>(
    h_expected, input);
  test_load_direct_blocked_vectorized_non_default_constructible<modifier, 2, items_per_thread, threads_in_block>(
    h_expected, input);
  test_load_direct_blocked_vectorized_non_default_constructible<modifier, 4, items_per_thread, threads_in_block>(
    h_expected, input);
}
#endif
