// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <cub/detail/rebase_iterator.cuh>
#include <cub/iterator/cache_modified_input_iterator.cuh>

#include <thrust/functional.h>
#include <thrust/iterator/detail/tagged_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_input_output_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/system/cuda/pointer.h>
#include <thrust/zip_function.h>

#include <cuda/__container/heterogeneous_iterator.h>
#include <cuda/cmath>
#include <cuda/iterator>
#include <cuda/std/__iterator/bounded_iter.h>
#include <cuda/std/array>
#include <cuda/std/functional>
#include <cuda/std/iterator>
#include <cuda/std/span>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>

#include "cub_test_macros.h"

template <typename T, typename Array>
__device__ void load_array(const T* ptr, Array& regs)
{
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int k = 0; k < static_cast<int>(regs.size()); ++k)
  {
    regs[k] = ptr[k];
  }
}

template <typename Ptrs, typename Regs, cuda::std::size_t... Is>
__device__ void load_to_registers(const Ptrs& ptrs, Regs& regs, cuda::std::index_sequence<Is...>)
{
  (load_array(cuda::std::get<Is>(ptrs), cuda::std::get<Is>(regs)), ...);
}

template <typename... Arrays, cuda::std::size_t... Is>
__device__ auto data_pointers(cuda::std::tuple<Arrays...>& regs, cuda::std::index_sequence<Is...>)
{
  return cuda::std::tuple<typename Arrays::value_type*...>{cuda::std::get<Is>(regs).data()...};
}

template <int ItemsPerThread, typename... Ts>
__device__ auto make_registers(const cuda::std::tuple<Ts*...>&)
{
  return cuda::std::tuple<cuda::std::array<cuda::std::remove_const_t<Ts>, ItemsPerThread>...>{};
}

// Each thread loads ItemsPerThread consecutive items twice: once directly through the iterator, and once by loading the
// iterator's raw pointers into registers and reading through the iterator rebased onto these registers.
template <int ItemsPerThread, typename It>
__global__ void load_kernel(It it, int num_threads)
{
  using value_t = cuda::std::iter_value_t<It>;
  const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (tid >= num_threads)
  {
    return;
  }

  const It thread_it = it + tid * ItemsPerThread;

  // load directly through the iterator
  value_t direct_items[ItemsPerThread];
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < ItemsPerThread; ++i)
    {
      direct_items[i] = thread_it[i];
    }
  }

  // load through the rebased iterator
  value_t rebased_items[ItemsPerThread];
  {
    // load the raw pointers' data into registers and rebase the iterator onto them
    const auto ptrs        = cub::detail::extract_raw_pointers(thread_it);
    constexpr auto indices = cuda::std::make_index_sequence<cuda::std::tuple_size_v<decltype(ptrs)>>{};
    auto regs              = make_registers<ItemsPerThread>(ptrs);
    load_to_registers(ptrs, regs, indices);
    const It rebased_it = cub::detail::rebase_iterator(thread_it, data_pointers(regs, indices));

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < ItemsPerThread; ++i)
    {
      rebased_items[i] = rebased_it[i];
    }
  }

  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < ItemsPerThread; ++i)
  {
    REQUIRE(direct_items[i] == rebased_items[i]);
  }
}

constexpr int num_threads = 3 * 128 + 5;
constexpr int max_items   = num_threads * 7;

template <int ItemsPerThread, typename It>
void check_rebase(It it)
{
  constexpr int block_threads = 128;
  load_kernel<ItemsPerThread><<<cuda::ceil_div(num_threads, block_threads), block_threads>>>(it, num_threads);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
}

template <typename It>
void check_rebase(It it)
{
  check_rebase<1>(it);
  check_rebase<4>(it);
  check_rebase<7>(it);
}

template <typename Ptrs, typename Regs, cuda::std::size_t... Is>
__device__ void require_equal(const Ptrs& ptrs, const Regs& regs, cuda::std::index_sequence<Is...>)
{
  (
    [&] {
      for (int i = 0; i < static_cast<int>(cuda::std::get<Is>(regs).size()); ++i)
      {
        REQUIRE(cuda::std::get<Is>(regs)[i] == cuda::std::get<Is>(ptrs)[i]);
      }
    }(),
    ...);
}

// Each thread stores ItemsPerThread consecutive items twice: once directly through the iterator, and once through the
// iterator rebased onto registers. The registers must then hold the same values as the iterator's raw pointers.
template <int ItemsPerThread, typename It, typename Gen>
__global__ void store_kernel(It it, Gen gen, int num_threads)
{
  const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (tid >= num_threads)
  {
    return;
  }

  const It thread_it = it + tid * ItemsPerThread;

  // store directly through the iterator
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < ItemsPerThread; ++i)
  {
    thread_it[i] = gen(tid * ItemsPerThread + i);
  }

  // rebase the iterator onto registers
  const auto ptrs        = cub::detail::extract_raw_pointers(thread_it);
  constexpr auto indices = cuda::std::make_index_sequence<cuda::std::tuple_size_v<decltype(ptrs)>>{};
  auto regs              = make_registers<ItemsPerThread>(ptrs);
  const It rebased_it    = cub::detail::rebase_iterator(thread_it, data_pointers(regs, indices));

  // store through the rebased iterator
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < ItemsPerThread; ++i)
  {
    rebased_it[i] = gen(tid * ItemsPerThread + i);
  }

  require_equal(ptrs, regs, indices);
}

template <int ItemsPerThread, typename It, typename Gen>
void check_store(It it, Gen gen)
{
  constexpr int block_threads = 128;
  store_kernel<ItemsPerThread><<<cuda::ceil_div(num_threads, block_threads), block_threads>>>(it, gen, num_threads);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
}

template <typename It, typename Gen = cuda::std::identity>
void check_store(It it, Gen gen = {})
{
  check_store<1>(it, gen);
  check_store<4>(it, gen);
  check_store<7>(it, gen);
}

// Rebasing onto other pointers on the host must produce an iterator of the same type that unwraps to these pointers.
template <typename It, typename... Ts>
void check_round_trip(It it, cuda::std::tuple<Ts*...> new_ptrs)
{
  STATIC_REQUIRE(cub::detail::raw_pointer_count_v<It> == sizeof...(Ts));
  const auto rebased = cub::detail::rebase_iterator(it, new_ptrs);
  STATIC_REQUIRE(cuda::std::is_same_v<decltype(rebased), const It>);
  CHECK(cub::detail::extract_raw_pointers(rebased) == cub::detail::raw_pointers_t<It>{new_ptrs});
}

template <typename T>
c2h::device_vector<T> make_sequence(int size = max_items)
{
  c2h::device_vector<T> v(size);
  thrust::sequence(v.begin(), v.end(), T{1});
  return v;
}

CUB_TEST("rebase_iterator handles contiguous iterators", "[util][iterator]", CUB_SMALL)
{
  auto ints = make_sequence<int>();
  int other[1]{};

  int* ptr = thrust::raw_pointer_cast(ints.data());
  CHECK(cub::detail::extract_raw_pointers(ptr) == cuda::std::tuple<int*>{ptr});
  check_round_trip(ptr, cuda::std::tuple<int*>{other});
  check_rebase(ptr);

  const int* cptr = ptr;
  check_round_trip(cptr, cuda::std::tuple<int*>{other});
  check_rebase(cptr);

  CHECK(cub::detail::extract_raw_pointers(ints.begin()) == cuda::std::tuple<int*>{ptr});
  check_round_trip(ints.begin(), cuda::std::tuple<int*>{other});
  check_rebase(ints.begin());

  check_round_trip(ints.cbegin(), cuda::std::tuple<int*>{other});
  check_rebase(ints.cbegin());
}

CUB_TEST("rebase_iterator rebuilds all CCCL contiguous iterators", "[util][iterator]", CUB_SMALL)
{
  auto ints = make_sequence<int>();
  int* ptr  = thrust::raw_pointer_cast(ints.data());
  int other[1]{};
  const auto other_ptrs = cuda::std::tuple<int*>{other};

  const auto span = cuda::std::span<int>{ptr, max_items};
  check_round_trip(span.begin(), other_ptrs);
  check_rebase(span.begin());
  check_round_trip(cuda::std::span<const int>{span}.begin(), other_ptrs);
  check_rebase(cuda::std::span<const int>{span}.begin());

  // the bounds move along with the pointer
  const auto bounded = cuda::std::__make_bounded_iter(ptr, ptr, ptr + max_items);
  check_round_trip(bounded, other_ptrs);
  check_rebase(bounded);

  check_round_trip(thrust::device_ptr<int>{ptr}, other_ptrs);
  check_rebase(thrust::device_ptr<int>{ptr});

  check_round_trip(thrust::cuda::pointer<int>{ptr}, other_ptrs);
  check_rebase(thrust::cuda::pointer<int>{ptr});

  using tagged_t = thrust::detail::tagged_iterator<int*, thrust::cuda::tag>;
  check_round_trip(tagged_t{ptr}, other_ptrs);
  check_rebase(tagged_t{ptr});

  using heterogeneous_t = cuda::heterogeneous_iterator<int, cuda::mr::device_accessible>;
  check_round_trip(heterogeneous_t{ptr}, other_ptrs);
  check_rebase(heterogeneous_t{ptr});

  c2h::host_vector<int> host_ints(1);
  check_round_trip(host_ints.begin(), other_ptrs);
  check_round_trip(host_ints.cbegin(), other_ptrs);
}

CUB_TEST("rebase_iterator keeps opaque iterators", "[util][iterator]", CUB_SMALL)
{
  auto ints = make_sequence<int>(max_items * 3);
  int* ptr  = thrust::raw_pointer_cast(ints.data());

  check_round_trip(cuda::counting_iterator<int>{42}, cuda::std::tuple<>{});
  check_rebase(cuda::counting_iterator<int>{42});

  check_round_trip(cuda::constant_iterator<int>{42}, cuda::std::tuple<>{});
  check_rebase(cuda::constant_iterator<int>{42});

  // strided iterators are not contiguous
  check_round_trip(cuda::strided_iterator{ptr, 3}, cuda::std::tuple<>{});
  check_rebase(cuda::strided_iterator{ptr, 3});

  // reverse iterators hold a pointer, but traverse the storage backwards
  const auto reverse_it = cuda::std::reverse_iterator{ptr + max_items};
  check_round_trip(reverse_it, cuda::std::tuple<>{});
  check_rebase(reverse_it);
  STATIC_REQUIRE(
    cub::detail::raw_pointer_count_v<thrust::detail::tagged_iterator<decltype(reverse_it), thrust::cuda::tag>> == 0);

  // the cache modifier would be applied to the rebased pointer, so the iterator must not be unwrapped
  STATIC_REQUIRE(cub::detail::raw_pointer_count_v<cub::CacheModifiedInputIterator<cub::LOAD_CG, int>> == 0);
}

CUB_TEST("rebase_iterator handles transform iterators", "[util][iterator]", CUB_SMALL)
{
  auto ints = make_sequence<int>();
  int* ptr  = thrust::raw_pointer_cast(ints.data());
  int other[1]{};

  const auto cuda_it = cuda::transform_iterator{ptr, thrust::square<>{}};
  CHECK(cub::detail::extract_raw_pointers(cuda_it) == cuda::std::tuple<int*>{ptr});
  check_round_trip(cuda_it, cuda::std::tuple<int*>{other});
  check_rebase(cuda_it);

  const auto thrust_it = thrust::make_transform_iterator(ints.cbegin(), cuda::std::negate<>{});
  check_round_trip(thrust_it, cuda::std::tuple<int*>{other});
  check_rebase(thrust_it);

  // transform of an opaque iterator
  const auto counting_it = cuda::transform_iterator{cuda::counting_iterator<int>{3}, thrust::square<>{}};
  check_round_trip(counting_it, cuda::std::tuple<>{});
  check_rebase(counting_it);
}

CUB_TEST("rebase_iterator only rebases the index of permutation iterators", "[util][iterator]", CUB_SMALL)
{
  auto values  = make_sequence<float>();
  auto indices = make_sequence<int>();
  thrust::sequence(indices.begin(), indices.end(), max_items - 1, -1); // reversed

  float* values_ptr = thrust::raw_pointer_cast(values.data());
  int* indices_ptr  = thrust::raw_pointer_cast(indices.data());
  int other[1]{};

  const auto it = cuda::permutation_iterator{values_ptr, indices_ptr};
  CHECK(cub::detail::extract_raw_pointers(it) == cuda::std::tuple<int*>{indices_ptr});
  check_round_trip(it, cuda::std::tuple<int*>{other});
  check_rebase(it);
}

CUB_TEST("rebase_iterator handles nested zip iterators", "[util][iterator]", CUB_SMALL)
{
  auto floats = make_sequence<float>();
  auto ints   = make_sequence<int>();
  auto longs  = make_sequence<long>();

  const float* floats_ptr = thrust::raw_pointer_cast(floats.data());
  int* ints_ptr           = thrust::raw_pointer_cast(ints.data());
  long* longs_ptr         = thrust::raw_pointer_cast(longs.data());
  float other_floats[1]{};
  int other_ints[1]{};
  long other_longs[1]{};

  // zip(transform(float*), zip(int*, counting), constant, thrust::transform(device_vector<long>::iterator))
  const auto it = cuda::zip_iterator{
    cuda::transform_iterator{floats_ptr, thrust::square<>{}},
    cuda::zip_iterator{ints_ptr, cuda::counting_iterator<int>{7}},
    cuda::constant_iterator<short>{3},
    thrust::make_transform_iterator(longs.begin(), cuda::std::negate<>{})};

  // pointers are extracted depth first
  CHECK(cub::detail::extract_raw_pointers(it)
        == cuda::std::tuple<const float*, int*, long*>{floats_ptr, ints_ptr, longs_ptr});
  check_round_trip(it, cuda::std::tuple<float*, int*, long*>{other_floats, other_ints, other_longs});
  check_rebase(it);
}

CUB_TEST("rebase_iterator handles zip transform iterators", "[util][iterator]", CUB_SMALL)
{
  auto ints    = make_sequence<int>(max_items * 2);
  auto doubles = make_sequence<double>();

  int* ints_ptr       = thrust::raw_pointer_cast(ints.data());
  double* doubles_ptr = thrust::raw_pointer_cast(doubles.data());
  int other_ints[1]{};
  double other_doubles[1]{};

  using namespace thrust::placeholders;
  const auto it = cuda::zip_transform_iterator{
    _1 + 10.0 * _2,
    cuda::zip_transform_iterator{cuda::std::plus<>{}, ints_ptr, cuda::strided_iterator{ints_ptr, 2}},
    doubles_ptr};
  CHECK(cub::detail::extract_raw_pointers(it) == cuda::std::tuple<int*, double*>{ints_ptr, doubles_ptr});
  check_round_trip(it, cuda::std::tuple<int*, double*>{other_ints, other_doubles});
  check_rebase(it);
}

CUB_TEST("rebase_iterator handles thrust zip iterators", "[util][iterator]", CUB_SMALL)
{
  auto ints    = make_sequence<int>();
  auto doubles = make_sequence<double>();
  int other_ints[1]{};
  double other_doubles[1]{};

  const auto it = thrust::make_zip_iterator(
    cuda::transform_iterator{thrust::raw_pointer_cast(ints.data()), thrust::square<>{}},
    doubles.cbegin(),
    cuda::counting_iterator<int>{0});
  check_round_trip(it, cuda::std::tuple<int*, double*>{other_ints, other_doubles});
  check_rebase(it);
}

CUB_TEST("rebase_iterator handles a deep mix of cuda and thrust iterators", "[util][iterator]", CUB_SMALL)
{
  auto ints      = make_sequence<int>();
  auto indices   = make_sequence<int>();
  auto floats    = make_sequence<float>();
  auto doubles   = make_sequence<double>();
  auto longs     = make_sequence<long>(max_items * 2);
  auto shorts    = make_sequence<short>();
  auto uchars    = make_sequence<unsigned char>();
  auto unsigneds = make_sequence<unsigned>();
  thrust::sequence(indices.begin(), indices.end(), max_items - 1, -1); // reversed

  int* ints_ptr             = thrust::raw_pointer_cast(ints.data());
  const int* indices_ptr    = thrust::raw_pointer_cast(indices.data());
  const double* doubles_ptr = thrust::raw_pointer_cast(doubles.data());
  short* shorts_ptr         = thrust::raw_pointer_cast(shorts.data());
  unsigned char* uchars_ptr = thrust::raw_pointer_cast(uchars.data());
  unsigned* unsigneds_ptr   = thrust::raw_pointer_cast(unsigneds.data());

  // thrust::zip(
  //   cuda::zip(
  //     cuda::transform(thrust::zip(device_vector<int>::iterator, counting), cuda::zip_function(plus)),
  //     constant,
  //     thrust::transform(cuda::permutation(float*, device_vector<int>::const_iterator), negate)),
  //   cuda::zip_transform(_1 + _2 * _3,
  //     cuda::transform(cuda::zip(device_vector<double>::const_iterator, constant), thrust::zip_function(multiplies)),
  //     strided(long*),
  //     thrust::transform(short*, thrust::square)),
  //   counting,
  //   thrust::transform(
  //     cuda::zip(unsigned char*,
  //               counting,
  //               cuda::transform(cuda::zip(constant, device_vector<unsigned>::iterator),
  //               cuda::zip_function(multiplies))),
  //     thrust::zip_function(_1 * _2 - _3)))
  using namespace thrust::placeholders;
  const auto a = cuda::zip_iterator{
    cuda::transform_iterator{thrust::make_zip_iterator(ints.begin(), cuda::counting_iterator<int>{5}),
                             cuda::zip_function{cuda::std::plus<>{}}},
    cuda::constant_iterator<char>{'x'},
    thrust::make_transform_iterator(
      cuda::permutation_iterator{thrust::raw_pointer_cast(floats.data()), indices.cbegin()}, cuda::std::negate<>{})};
  const auto b = cuda::zip_transform_iterator{
    _1 + _2 * _3,
    cuda::transform_iterator{cuda::zip_iterator{doubles.cbegin(), cuda::constant_iterator<int>{2}},
                             thrust::make_zip_function(cuda::std::multiplies<>{})},
    cuda::strided_iterator{thrust::raw_pointer_cast(longs.data()), 2},
    thrust::make_transform_iterator(shorts_ptr, thrust::square<>{})};
  const auto c = cuda::counting_iterator<long long>{100};
  const auto d = thrust::make_transform_iterator(
    cuda::zip_iterator{
      uchars_ptr,
      cuda::counting_iterator<int>{-3},
      cuda::transform_iterator{cuda::zip_iterator{cuda::constant_iterator<float>{1.5f}, unsigneds.begin()},
                               cuda::zip_function{cuda::std::multiplies<>{}}}},
    thrust::make_zip_function(_1 * _2 - _3));
  const auto it = thrust::make_zip_iterator(a, b, c, d);

  using ptrs_t = cuda::std::tuple<int*, const int*, const double*, short*, unsigned char*, unsigned*>;
  CHECK(cub::detail::extract_raw_pointers(it)
        == ptrs_t{ints_ptr, indices_ptr, doubles_ptr, shorts_ptr, uchars_ptr, unsigneds_ptr});

  int other_ints[1]{};
  double other_doubles[1]{};
  short other_shorts[1]{};
  unsigned char other_uchars[1]{};
  unsigned other_unsigneds[1]{};
  check_round_trip(it,
                   cuda::std::tuple<int*, int*, double*, short*, unsigned char*, unsigned*>{
                     other_ints, other_ints, other_doubles, other_shorts, other_uchars, other_unsigneds});
  check_rebase(it);
}

CUB_TEST("rebase_iterator handles move and counted iterators", "[util][iterator]", CUB_SMALL)
{
  auto ints = make_sequence<int>();
  int* ptr  = thrust::raw_pointer_cast(ints.data());
  int other[1]{};
  const auto other_ptrs = cuda::std::tuple<int*>{other};

  const auto move_it = cuda::std::move_iterator{ptr};
  check_round_trip(move_it, other_ptrs);
  check_rebase(move_it);

  // counted iterators over a pointer are contiguous, but cannot be constructed from a pointer
  const auto counted_it = cuda::std::counted_iterator{ptr, max_items};
  check_round_trip(counted_it, other_ptrs);
  check_rebase(counted_it);

  const auto counted_transform_it =
    cuda::std::counted_iterator{cuda::transform_iterator{ptr, thrust::square<>{}}, max_items};
  check_round_trip(counted_transform_it, other_ptrs);
  check_rebase(counted_transform_it);
}

CUB_TEST("rebase_iterator only rebases the index of thrust permutation iterators", "[util][iterator]", CUB_SMALL)
{
  auto values  = make_sequence<float>();
  auto indices = make_sequence<int>();
  thrust::sequence(indices.begin(), indices.end(), max_items - 1, -1); // reversed
  int other[1]{};

  const auto it = thrust::make_permutation_iterator(values.cbegin(), indices.cbegin());
  CHECK(cub::detail::extract_raw_pointers(it)
        == cuda::std::tuple<const int*>{thrust::raw_pointer_cast(indices.data())});
  check_round_trip(it, cuda::std::tuple<int*>{other});
  check_rebase(it);
}

CUB_TEST("rebase_iterator handles output iterators", "[util][iterator]", CUB_SMALL)
{
  auto ints = make_sequence<int>();
  auto more = make_sequence<int>();
  int* ptr  = thrust::raw_pointer_cast(ints.data());
  int other[1]{};
  const auto other_ptrs = cuda::std::tuple<int*>{other};

  check_store(ptr);
  check_store(ints.begin());

  using namespace thrust::placeholders;
  const auto cuda_it = cuda::transform_output_iterator{ptr, cuda::std::negate<>{}};
  check_round_trip(cuda_it, other_ptrs);
  check_store(cuda_it);

  const auto thrust_it = thrust::make_transform_output_iterator(ints.begin(), _1 * 3);
  check_round_trip(thrust_it, other_ptrs);
  check_store(thrust_it);

  // transform input output iterators can be read and written
  const auto cuda_io_it = cuda::transform_input_output_iterator{ptr, thrust::square<>{}, _1 + 7};
  check_round_trip(cuda_io_it, other_ptrs);
  check_rebase(cuda_io_it);
  check_store(cuda_io_it);

  const auto thrust_io_it =
    thrust::make_transform_input_output_iterator(ints.begin(), cuda::std::negate<>{}, thrust::square<>{});
  check_round_trip(thrust_io_it, other_ptrs);
  check_rebase(thrust_io_it);
  check_store(thrust_io_it);

  // nested output iterators
  const auto nested_it = cuda::transform_output_iterator{
    thrust::make_transform_output_iterator(cuda::transform_output_iterator{ptr, _1 - 1}, thrust::square<>{}),
    cuda::std::negate<>{}};
  check_round_trip(nested_it, other_ptrs);
  check_store(nested_it);
}
