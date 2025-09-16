/******************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
// #define CCCL_CHECK_SASS

#if defined(CCCL_CHECK_SASS)

#  include <cub/detail/type_traits.cuh>
#  include <cub/thread/thread_reduce.cuh>
#  include <cub/util_macro.cuh>

#  include <thrust/iterator/constant_iterator.h>

#  include <cuda/functional>
#  include <cuda/std/functional>
#  include <cuda/std/limits>
#  include <cuda/std/type_traits>

#  include <cstring>
#  include <functional>
#  include <limits>
#  include <numeric>

#  include "c2h/catch2_test_helper.h"
#  include "c2h/extended_types.h"
#  include "c2h/generators.h"
#  include <catch2/matchers/catch_matchers_floating_point.hpp>

/***********************************************************************************************************************
 * Thread Reduce Wrapper Kernels
 **********************************************************************************************************************/

template <int NUM_ITEMS, typename T, typename ReduceOperator>
__global__ void thread_reduce_kernel(const T* __restrict__ d_in, T* __restrict__ d_out, ReduceOperator reduce_operator)
{
  T thread_data[NUM_ITEMS];
  auto d_in_aligned = static_cast<T*>(__builtin_assume_aligned(d_in, (sizeof(T) < 4) ? 4 : sizeof(T)));
  ::memcpy(thread_data, d_in_aligned, sizeof(thread_data));
  *d_out = cub::ThreadReduce(thread_data, reduce_operator);
}

template <int NUM_ITEMS, typename T, typename ReduceOperator>
__global__ void thread_reduce_kernel_array(const T* d_in, T* d_out, ReduceOperator reduce_operator)
{
  cuda::std::array<T, NUM_ITEMS> thread_data;
#  pragma unroll
  for (int i = 0; i < NUM_ITEMS; ++i)
  {
    thread_data[i] = d_in[i];
  }
  *d_out = cub::ThreadReduce(thread_data, reduce_operator);
}

/***********************************************************************************************************************
 * CUB operator to STD operator
 **********************************************************************************************************************/

template <typename T, typename>
struct cub_operator_to_std;

template <typename T>
struct cub_operator_to_std<T, cuda::std::plus<>>
{
  using type = ::std::plus<>;
};

template <typename T>
struct cub_operator_to_std<T, cuda::std::multiplies<>>
{
  using type = ::std::multiplies<>;
};

template <typename T>
struct cub_operator_to_std<T, cuda::std::bit_xor<>>
{
  using type = ::std::bit_xor<>;
};

template <typename T>
struct cub_operator_to_std<T, cuda::minimum<>>
{
  using type = cuda::minimum<>;
};

template <typename T, typename Operator>
using cub_operator_to_std_t = typename cub_operator_to_std<T, Operator>::type;

/***********************************************************************************************************************
 * CUB operator to identity
 **********************************************************************************************************************/

template <typename T, typename Operator, typename = void>
struct cub_operator_to_identity;

template <typename T>
struct cub_operator_to_identity<T, cuda::std::plus<>>
{
  static constexpr T value()
  {
    return T{};
  }
};

template <typename T>
struct cub_operator_to_identity<T, cuda::std::multiplies<>>
{
  static constexpr T value()
  {
    return T{1};
  }
};

template <typename T>
struct cub_operator_to_identity<T, cuda::std::bit_xor<>>
{
  static constexpr T value()
  {
    return T{0};
  }
};

template <typename T>
struct cub_operator_to_identity<T, cuda::minimum<>>
{
  static constexpr T value()
  {
    return ::std::numeric_limits<T>::max();
  }
};

/***********************************************************************************************************************
 * Type list definition
 **********************************************************************************************************************/

using narrow_precision_type_list = c2h::type_list<
#  if TEST_HALF_T()
  __half,
#  endif // TEST_HALF_T()
#  if TEST_BF_T()
  __nv_bfloat16
#  endif // TEST_BF_T()
  >;

using fp_type_list = c2h::type_list<float>;

using integral_type_list = c2h::type_list<cuda::std::int8_t, cuda::std::int16_t, cuda::std::int32_t>;

using cub_operator_integral_list =
  c2h::type_list<cuda::std::plus<>, cuda::std::multiplies<>, cuda::std::bit_xor<>, cuda::minimum<>>;

using cub_operator_fp_list = c2h::type_list<cuda::std::plus<>, cuda::minimum<>>;

/***********************************************************************************************************************
 * Verify results and kernel launch
 **********************************************************************************************************************/

_CCCL_TEMPLATE(typename T)
_CCCL_REQUIRES((cuda::std::is_floating_point<T>::value))
void verify_results(const T& expected_data, const T& test_results)
{
  REQUIRE_THAT(expected_data, Catch::Matchers::WithinRel(test_results, T{0.05}));
}

_CCCL_TEMPLATE(typename T)
_CCCL_REQUIRES((!cuda::std::is_floating_point<T>::value))
void verify_results(const T& expected_data, const T& test_results)
{
  REQUIRE(expected_data == test_results);
}

template <typename T, typename ReduceOperator>
void run_thread_reduce_kernel(
  const c2h::device_vector<T>& in, c2h::device_vector<T>& out, ReduceOperator reduce_operator)
{
  thread_reduce_kernel<18>
    <<<1, 1>>>(thrust::raw_pointer_cast(in.data()), thrust::raw_pointer_cast(out.data()), reduce_operator);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
}

/***********************************************************************************************************************
 * Test cases
 **********************************************************************************************************************/

constexpr int size = 16;

C2H_TEST("ThreadReduce Integral Type Tests", "[reduce][thread]", integral_type_list, cub_operator_integral_list)
{
  using value_t                    = c2h::get<0, TestType>;
  constexpr auto reduce_op         = c2h::get<1, TestType>{};
  constexpr auto std_reduce_op     = cub_operator_to_std_t<value_t, c2h::get<1, TestType>>{};
  constexpr auto operator_identity = cub_operator_to_identity<value_t, c2h::get<1, TestType>>::value();
  CAPTURE(c2h::type_name<value_t>(), size, c2h::type_name<decltype(reduce_op)>());
  c2h::device_vector<value_t> d_in(size);
  c2h::device_vector<value_t> d_out(1);
  c2h::gen(C2H_SEED(1), d_in, std::numeric_limits<value_t>::min());
  c2h::host_vector<value_t> h_in = d_in;
  auto reference_result          = std::accumulate(h_in.begin(), h_in.begin() + size, operator_identity, std_reduce_op);
  run_thread_reduce_kernel(d_in, d_out, reduce_op);
  verify_results(reference_result, c2h::host_vector<value_t>(d_out)[0]);
}

C2H_TEST("ThreadReduce Floating-Point Type Tests", "[reduce][thread]", fp_type_list, cub_operator_fp_list)
{
  using value_t                = c2h::get<0, TestType>;
  constexpr auto reduce_op     = c2h::get<1, TestType>{};
  constexpr auto std_reduce_op = cub_operator_to_std_t<value_t, c2h::get<1, TestType>>{};
  const auto operator_identity = cub_operator_to_identity<value_t, c2h::get<1, TestType>>::value();
  CAPTURE(c2h::type_name<value_t>(), size, c2h::type_name<decltype(reduce_op)>());
  c2h::device_vector<value_t> d_in(size);
  c2h::device_vector<value_t> d_out(1);
  c2h::gen(C2H_SEED(1), d_in, std::numeric_limits<value_t>::min());
  c2h::host_vector<value_t> h_in = d_in;
  auto reference_result          = std::accumulate(h_in.begin(), h_in.begin() + size, operator_identity, std_reduce_op);
  run_thread_reduce_kernel(d_in, d_out, reduce_op);
  verify_results(reference_result, c2h::host_vector<value_t>(d_out)[0]);
}

#  if TEST_HALF_T() || TEST_BF_T()

C2H_TEST("ThreadReduce Narrow PrecisionType Tests",
         "[reduce][thread][narrow]",
         narrow_precision_type_list,
         cub_operator_fp_list)
{
  using value_t                = c2h::get<0, TestType>;
  constexpr auto reduce_op     = c2h::get<1, TestType>{};
  constexpr auto std_reduce_op = cub_operator_to_std_t<float, c2h::get<1, TestType>>{};
  const auto operator_identity = cub_operator_to_identity<float, c2h::get<1, TestType>>::value();
  c2h::device_vector<value_t> d_in(size);
  c2h::device_vector<value_t> d_out(1);
  c2h::gen(C2H_SEED(1), d_in, value_t{1.0f}, value_t{2.0f});
  c2h::host_vector<float> h_in_float = d_in;
  CAPTURE(c2h::type_name<value_t>(), size, c2h::type_name<decltype(reduce_op)>());
  auto reference_result =
    std::accumulate(h_in_float.begin(), h_in_float.begin() + size, operator_identity, std_reduce_op);
  run_thread_reduce_kernel(d_in, d_out, reduce_op);
  verify_results(reference_result, float{c2h::host_vector<value_t>(d_out)[0]});
}

#  endif // TEST_HALF_T() || TEST_BF_T()

#else

#  include "c2h/catch2_test_helper.h"

C2H_TEST("ThreadReduce Empty Test", "[reduce][thread][empty]") {}

#endif // CCCL_CHECK_SASS
