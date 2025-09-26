/******************************************************************************
 * Copyright (c) 2011-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include <cuda/iterator>

#include <nvbench_helper.cuh>

template <class InT, class OutT>
struct fib_t
{
  __device__ OutT operator()(InT n)
  {
    OutT t1 = 0;
    OutT t2 = 1;

    if (n < 1)
    {
      return t1;
    }
    else if (n == 1)
    {
      return t1;
    }
    else if (n == 2)
    {
      return t2;
    }
    for (InT i = 3; i <= n; ++i)
    {
      const auto next = t1 + t2;
      t1              = t2;
      t2              = next;
    }

    return t2;
  }
};

template <typename... Args>
void bench_copy(nvbench::state& state, Args&&... args)
{
  caching_allocator_t alloc; // copy shouldn't allocate, but let's be consistent
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch | nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) {
               thrust::copy(policy(alloc, launch), ::cuda::std::forward<Args>(args)...);
             });
}

template <typename T>
static void basic(nvbench::state& state, nvbench::type_list<T>)
{
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements"));

  thrust::device_vector<T> input = generate(elements, bit_entropy::_1_000, T{0}, T{42});
  thrust::device_vector<T> output(elements);

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements);
  state.add_global_memory_writes<T>(elements);

  fib_t<T, nvbench::uint32_t> op{};
  cuda::zip_transform_iterator first{op, input.cbegin()};
  cuda::zip_transform_iterator last{op, input.cend()};

  bench_copy(state, first, last, output.begin());
}

using types = nvbench::type_list<nvbench::uint32_t, nvbench::uint64_t>;

NVBENCH_BENCH_TYPES(basic, NVBENCH_TYPE_AXES(types))
  .set_name("base")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4));

namespace babelstream
{
// The benchmarks in this namespace are inspired by the BabelStream thrust version:
// https://github.com/UoB-HPC/BabelStream/blob/main/src/thrust/ThrustStream.cu

// Modified from BabelStream to also work for integers
constexpr auto startA      = 1; // BabelStream: 0.1
constexpr auto startB      = 2; // BabelStream: 0.2
constexpr auto startC      = 3; // BabelStream: 0.1
constexpr auto startScalar = 4; // BabelStream: 0.4

using element_types = nvbench::type_list<std::int8_t, std::int16_t, float, double, __int128>;
// Different benchmarks use a different number of buffers. H200/B200 can fit 2^31 elements for all benchmarks and types.
// Upstream BabelStream uses 2^25. Allocation failure just skips the benchmark
auto array_size_powers = std::vector<std::int64_t>{25, 31};

template <typename T>
static void mul(nvbench::state& state, nvbench::type_list<T>)
{
  const auto n = static_cast<std::size_t>(state.get_int64("Elements"));
  thrust::device_vector<T> b(n, startB);
  thrust::device_vector<T> c(n, startC);

  state.add_element_count(n);
  state.add_global_memory_reads<T>(n);
  state.add_global_memory_writes<T>(n);

  const T scalar = startScalar;
  auto op        = cuda::proclaim_copyable_arguments([=] _CCCL_DEVICE(const T& ci) {
    return ci * startScalar;
  });
  cuda::zip_transform_iterator first{op, c.begin()};
  cuda::zip_transform_iterator last{op, c.end()};
  bench_copy(state, first, last, b.begin());
}

NVBENCH_BENCH_TYPES(mul, NVBENCH_TYPE_AXES(element_types))
  .set_name("mul")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements", array_size_powers);

template <typename T>
static void add(nvbench::state& state, nvbench::type_list<T>)
{
  const auto n = static_cast<std::size_t>(state.get_int64("Elements"));
  thrust::device_vector<T> a(n, startA);
  thrust::device_vector<T> b(n, startB);
  thrust::device_vector<T> c(n, startC);

  state.add_element_count(n);
  state.add_global_memory_reads<T>(2 * n);
  state.add_global_memory_writes<T>(n);

  auto op = cuda::proclaim_copyable_arguments([] _CCCL_DEVICE(const T& ai, const T& bi) -> T {
    return ai + bi;
  });
  cuda::zip_transform_iterator first{op, a.begin(), b.begin()};
  cuda::zip_transform_iterator last{op, a.end(), b.end()};
  bench_copy(state, first, last, c.begin());
}

NVBENCH_BENCH_TYPES(add, NVBENCH_TYPE_AXES(element_types))
  .set_name("add")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements", array_size_powers);

template <typename T>
static void triad(nvbench::state& state, nvbench::type_list<T>)
{
  const auto n = static_cast<std::size_t>(state.get_int64("Elements"));
  thrust::device_vector<T> a(n, startA);
  thrust::device_vector<T> b(n, startB);
  thrust::device_vector<T> c(n, startC);

  state.add_element_count(n);
  state.add_global_memory_reads<T>(2 * n);
  state.add_global_memory_writes<T>(n);

  const T scalar = startScalar;
  auto op        = cuda::proclaim_copyable_arguments([=] _CCCL_DEVICE(const T& bi, const T& ci) {
    return bi + scalar * ci;
  });
  cuda::zip_transform_iterator first{op, b.begin(), c.begin()};
  cuda::zip_transform_iterator last{op, b.end(), c.end()};
  bench_copy(state, first, last, a.begin());
}

NVBENCH_BENCH_TYPES(triad, NVBENCH_TYPE_AXES(element_types))
  .set_name("triad")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements", array_size_powers);

template <typename T>
static void nstream(nvbench::state& state, nvbench::type_list<T>)
{
  const auto n = static_cast<std::size_t>(state.get_int64("Elements"));
  thrust::device_vector<T> a(n, startA);
  thrust::device_vector<T> b(n, startB);
  thrust::device_vector<T> c(n, startC);

  state.add_element_count(n);
  state.add_global_memory_reads<T>(3 * n);
  state.add_global_memory_writes<T>(n);

  const T scalar = startScalar;
  auto op        = cuda::proclaim_copyable_arguments([=] _CCCL_DEVICE(const T& ai, const T& bi, const T& ci) {
    return ai + bi + scalar * ci;
  });
  cuda::zip_transform_iterator first{op, a.begin(), b.begin(), c.begin()};
  cuda::zip_transform_iterator last{op, a.end(), b.end(), c.end()};
  bench_copy(state, first, last, a.begin());
}

NVBENCH_BENCH_TYPES(nstream, NVBENCH_TYPE_AXES(element_types))
  .set_name("nstream")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements", array_size_powers);
} // namespace babelstream
