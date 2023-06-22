/******************************************************************************
 * Copyright (c) 2011-2021, NVIDIA CORPORATION.  All rights reserved.
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

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <cub/device/device_adjacent_difference.cuh>
#include <cub/thread/thread_operators.cuh>
#include <cub/util_allocator.cuh>

#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>

#include <limits>
#include <memory>

#include "test_util.h"


using namespace cub;


constexpr bool READ_LEFT = true;
constexpr bool READ_RIGHT = false;


/**
 * \brief Generates integer sequence \f$S_n=i(i-1)/2\f$.
 *
 * The adjacent difference of this sequence produce consecutive numbers:
 * \f[
 *   p = \frac{i(i - 1)}{2} \\
 *   n = \frac{(i + 1) i}{2} \\
 *   n - p = i \\
 *   \frac{(i + 1) i}{2} - \frac{i (i - 1)}{2} = i \\
 *   (i + 1) i - i (i - 1) = 2 i \\
 *   (i + 1) - (i - 1) = 2 \\
 *   2 = 2
 * \f]
 */
template <typename DestT>
struct TestSequenceGenerator
{
  template <typename SourceT>
  __device__ __host__ DestT operator()(SourceT index) const
  {
    return static_cast<DestT>(index * (index - 1) / SourceT(2));
  }
};


template <typename OutputT>
struct CustomDifference
{
  template <typename InputT>
  __device__ OutputT operator()(const InputT &lhs, const InputT &rhs)
  {
    return static_cast<OutputT>(lhs - rhs);
  }
};

template <bool ReadLeft,
          typename IteratorT,
          typename DifferenceOpT,
          typename NumItemsT>
void AdjacentDifference(void *temp_storage,
                        std::size_t &temp_storage_bytes,
                        IteratorT it,
                        DifferenceOpT difference_op,
                        NumItemsT num_items)
{
  const bool is_default_op_in_use =
    std::is_same<DifferenceOpT, cub::Difference>::value;

  if (ReadLeft)
  {
    if (is_default_op_in_use)
    {
      CubDebugExit(
        cub::DeviceAdjacentDifference::SubtractLeft(temp_storage,
                                                    temp_storage_bytes,
                                                    it,
                                                    num_items));
    }
    else
    {
      CubDebugExit(
        cub::DeviceAdjacentDifference::SubtractLeft(temp_storage,
                                                    temp_storage_bytes,
                                                    it,
                                                    num_items,
                                                    difference_op));
    }
  }
  else
  {
    if (is_default_op_in_use)
    {
      CubDebugExit(
        cub::DeviceAdjacentDifference::SubtractRight(temp_storage,
                                                     temp_storage_bytes,
                                                     it,
                                                     num_items));
    }
    else
    {
      CubDebugExit(
        cub::DeviceAdjacentDifference::SubtractRight(temp_storage,
                                                     temp_storage_bytes,
                                                     it,
                                                     num_items,
                                                     difference_op));
    }
  }
}


template <bool ReadLeft,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename DifferenceOpT,
          typename NumItemsT>
void AdjacentDifferenceCopy(void *temp_storage,
                            std::size_t &temp_storage_bytes,
                            InputIteratorT input,
                            OutputIteratorT output,
                            DifferenceOpT difference_op,
                            NumItemsT num_items)
{
  const bool is_default_op_in_use =
    std::is_same<DifferenceOpT, cub::Difference>::value;

  if (ReadLeft)
  {
    if (is_default_op_in_use)
    {
      CubDebugExit(
        cub::DeviceAdjacentDifference::SubtractLeftCopy(temp_storage,
                                                        temp_storage_bytes,
                                                        input,
                                                        output,
                                                        num_items));
    }
    else
    {
      CubDebugExit(
        cub::DeviceAdjacentDifference::SubtractLeftCopy(temp_storage,
                                                        temp_storage_bytes,
                                                        input,
                                                        output,
                                                        num_items,
                                                        difference_op));
    }
  }
  else
  {
    if (is_default_op_in_use)
    {
      CubDebugExit(
        cub::DeviceAdjacentDifference::SubtractRightCopy(temp_storage,
                                                         temp_storage_bytes,
                                                         input,
                                                         output,
                                                         num_items));
    }
    else
    {
      CubDebugExit(
        cub::DeviceAdjacentDifference::SubtractRightCopy(temp_storage,
                                                         temp_storage_bytes,
                                                         input,
                                                         output,
                                                         num_items,
                                                         difference_op));
    }
  }
}

template <bool ReadLeft,
          typename IteratorT,
          typename DifferenceOpT,
          typename NumItemsT>
void AdjacentDifference(IteratorT it,
                        DifferenceOpT difference_op,
                        NumItemsT num_items)
{
  std::size_t temp_storage_bytes {};

  AdjacentDifference<ReadLeft>(nullptr,
                               temp_storage_bytes,
                               it,
                               difference_op,
                               num_items);

  thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  AdjacentDifference<ReadLeft>(thrust::raw_pointer_cast(temp_storage.data()),
                               temp_storage_bytes,
                               it,
                               difference_op,
                               num_items);
}


template <bool ReadLeft,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename DifferenceOpT,
          typename NumItemsT>
void AdjacentDifferenceCopy(InputIteratorT input,
                            OutputIteratorT output,
                            DifferenceOpT difference_op,
                            NumItemsT num_items)
{
  std::size_t temp_storage_bytes{};

  AdjacentDifferenceCopy<ReadLeft>(nullptr,
                                   temp_storage_bytes,
                                   input,
                                   output,
                                   difference_op,
                                   num_items);

  thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  AdjacentDifferenceCopy<ReadLeft>(thrust::raw_pointer_cast(
                                     temp_storage.data()),
                                   temp_storage_bytes,
                                   input,
                                   output,
                                   difference_op,
                                   num_items);
}

template <typename FirstIteratorT,
          typename SecondOperatorT>
bool CheckResult(FirstIteratorT first_begin,
                 FirstIteratorT first_end,
                 SecondOperatorT second_begin)
{
  auto err = thrust::mismatch(first_begin, first_end, second_begin);

  if (err.first != first_end)
  {
    return false;
  }

  return true;
}


template <typename InputT,
          typename OutputT,
          typename DifferenceOpT,
          typename NumItemsT>
void TestCopy(NumItemsT elements, DifferenceOpT difference_op)
{
  thrust::device_vector<InputT> input(elements);
  thrust::tabulate(input.begin(),
                   input.end(),
                   TestSequenceGenerator<InputT>{});

  thrust::device_vector<OutputT> output(elements, OutputT{42});

  InputT *d_input = thrust::raw_pointer_cast(input.data());
  OutputT *d_output = thrust::raw_pointer_cast(output.data());

  using CountingIteratorT =
    typename thrust::counting_iterator<OutputT,
                                       thrust::use_default,
                                       std::size_t,
                                       std::size_t>;

  AdjacentDifferenceCopy<READ_LEFT>(d_input,
                                    d_output,
                                    difference_op,
                                    elements);

  AssertTrue(CheckResult(output.begin() + 1,
                         output.end(),
                         CountingIteratorT(OutputT{0})));

  thrust::fill(output.begin(), output.end(), OutputT{42});

  AdjacentDifferenceCopy<READ_RIGHT>(d_input,
                                     d_output,
                                     difference_op,
                                     elements);

  thrust::device_vector<OutputT> reference(input.size());
  thrust::sequence(reference.begin(),
                   reference.end(),
                   static_cast<OutputT>(0),
                   static_cast<OutputT>(-1));
  AssertTrue(CheckResult(output.begin(),
                         output.end() - 1,
                         reference.begin()));
}


template <typename InputT,
          typename OutputT,
          typename DifferenceOpT,
          typename NumItemsT>
void TestIteratorCopy(NumItemsT elements, DifferenceOpT difference_op)
{
  thrust::device_vector<InputT> input(elements);
  thrust::tabulate(input.begin(),
                   input.end(),
                   TestSequenceGenerator<InputT>{});

  thrust::device_vector<OutputT> output(elements, OutputT{42});

  using CountingIteratorT =
  typename thrust::counting_iterator<OutputT,
    thrust::use_default,
    std::size_t,
    std::size_t>;

  AdjacentDifferenceCopy<READ_LEFT>(input.cbegin(),
                                    output.begin(),
                                    difference_op,
                                    elements);

  AssertTrue(CheckResult(output.begin() + 1,
                         output.end(),
                         CountingIteratorT(OutputT{0})));

  thrust::fill(output.begin(), output.end(), OutputT{42});

  AdjacentDifferenceCopy<READ_RIGHT>(input.cbegin(),
                                     output.begin(),
                                     difference_op,
                                     elements);

  thrust::device_vector<OutputT> reference(input.size());
  thrust::sequence(reference.begin(),
                   reference.end(),
                   static_cast<OutputT>(0),
                   static_cast<OutputT>(-1));
  AssertTrue(CheckResult(output.begin(),
                         output.end() - 1,
                         reference.begin()));
}


template <typename InputT,
          typename OutputT,
          typename NumItemsT>
void TestCopy(NumItemsT elements)
{
  TestCopy<InputT, OutputT>(elements, cub::Difference{});
  TestCopy<InputT, OutputT>(elements, CustomDifference<OutputT>{});

  TestIteratorCopy<InputT, OutputT>(elements, cub::Difference{});
  TestIteratorCopy<InputT, OutputT>(elements, CustomDifference<OutputT>{});
}


template <typename NumItemsT>
void TestCopy(NumItemsT elements)
{
  TestCopy<std::uint64_t, std::int64_t >(elements);
  TestCopy<std::uint32_t, std::int32_t>(elements);
}


template <typename T,
          typename DifferenceOpT,
          typename NumItemsT>
void Test(NumItemsT elements, DifferenceOpT difference_op)
{
  thrust::device_vector<T> data(elements);
  thrust::tabulate(data.begin(),
                   data.end(),
                   TestSequenceGenerator<T>{});

  T *d_data = thrust::raw_pointer_cast(data.data());

  using CountingIteratorT =
    typename thrust::counting_iterator<T,
      thrust::use_default,
      std::size_t,
      std::size_t>;

  AdjacentDifference<READ_LEFT>(d_data,
                                difference_op,
                                elements);

  AssertTrue(CheckResult(data.begin() + 1,
                         data.end(),
                         CountingIteratorT(T{0})));


  thrust::tabulate(data.begin(),
                   data.end(),
                   TestSequenceGenerator<T>{});

  AdjacentDifference<READ_RIGHT>(d_data,
                                 difference_op,
                                 elements);

  thrust::device_vector<T> reference(data.size());
  thrust::sequence(reference.begin(),
                   reference.end(),
                   static_cast<T>(0),
                   static_cast<T>(-1));
  AssertTrue(CheckResult(data.begin(),
                         data.end() - 1,
                         reference.begin()));
}


template <typename T,
          typename DifferenceOpT,
          typename NumItemsT>
void TestIterators(NumItemsT elements, DifferenceOpT difference_op)
{
  thrust::device_vector<T> data(elements);
  thrust::tabulate(data.begin(),
                   data.end(),
                   TestSequenceGenerator<T>{});

  using CountingIteratorT =
  typename thrust::counting_iterator<T,
    thrust::use_default,
    std::size_t,
    std::size_t>;

  AdjacentDifference<READ_LEFT>(data.begin(),
                                difference_op,
                                elements);

  AssertTrue(CheckResult(data.begin() + 1,
                         data.end(),
                         CountingIteratorT(T{0})));


  thrust::tabulate(data.begin(),
                   data.end(),
                   TestSequenceGenerator<T>{});

  AdjacentDifference<READ_RIGHT>(data.begin(),
                                 difference_op,
                                 elements);

  thrust::device_vector<T> reference(data.size());
  thrust::sequence(reference.begin(),
                   reference.end(),
                   static_cast<T>(0),
                   static_cast<T>(-1));

  AssertTrue(CheckResult(data.begin(), data.end() - 1, reference.begin()));
}


template <typename T,
          typename NumItemsT>
void Test(NumItemsT elements)
{
  Test<T>(elements, cub::Difference{});
  Test<T>(elements, CustomDifference<T>{});

  TestIterators<T>(elements, cub::Difference{});
  TestIterators<T>(elements, CustomDifference<T>{});
}


template <typename NumItemsT>
void Test(NumItemsT elements)
{
  Test<std::int32_t, NumItemsT>(elements);
  Test<std::uint32_t, NumItemsT>(elements);
  Test<std::uint64_t, NumItemsT>(elements);
}


template <typename ValueT,
          typename NumItemsT>
void TestFancyIterators(NumItemsT elements)
{
  if (elements == 0)
  {
    return;
  }

  thrust::counting_iterator<ValueT> count_iter(ValueT{1});
  thrust::device_vector<ValueT> output(elements, ValueT{42});

  AdjacentDifferenceCopy<READ_LEFT>(count_iter,
                                    output.begin(),
                                    cub::Difference{},
                                    elements);
  AssertEquals(elements,
               static_cast<NumItemsT>(
                 thrust::count(output.begin(), output.end(), ValueT(1))));

  thrust::fill(output.begin(), output.end(), ValueT{});
  AdjacentDifferenceCopy<READ_RIGHT>(count_iter,
                                     output.begin(),
                                     cub::Difference{},
                                     elements);
  AssertEquals(elements - 1,
               static_cast<NumItemsT>(
                 thrust::count(output.begin(),
                               output.end() - 1,
                               static_cast<ValueT>(-1))));
  AssertEquals(output.back(), static_cast<ValueT>(elements));

  thrust::constant_iterator<ValueT> const_iter(ValueT{});

  AdjacentDifferenceCopy<READ_LEFT>(const_iter,
                                    output.begin(),
                                    cub::Difference{},
                                    elements);
  AssertEquals(elements,
               static_cast<NumItemsT>(
                 thrust::count(output.begin(), output.end(), ValueT{})));

  thrust::fill(output.begin(), output.end(), ValueT{});
  AdjacentDifferenceCopy<READ_RIGHT>(const_iter,
                                     output.begin(),
                                     cub::Difference{},
                                     elements);
  AssertEquals(elements,
               static_cast<NumItemsT>(
                 thrust::count(output.begin(), output.end(), ValueT{})));

  AdjacentDifferenceCopy<READ_LEFT>(const_iter,
                                    thrust::make_discard_iterator(),
                                    cub::Difference{},
                                    elements);

  AdjacentDifferenceCopy<READ_RIGHT>(const_iter,
                                     thrust::make_discard_iterator(),
                                     cub::Difference{},
                                     elements);
}


template <typename NumItemsT>
void TestFancyIterators(NumItemsT elements)
{
  TestFancyIterators<std::uint64_t, NumItemsT>(elements);
}


template <typename NumItemsT>
void TestSize(NumItemsT elements)
{
  Test(elements);
  TestCopy(elements);
  TestFancyIterators(elements);
}

struct DetectWrongDifference
{
  bool *flag;

  __host__ __device__ DetectWrongDifference operator++() const
  {
    return *this;
  }
  __host__ __device__ DetectWrongDifference operator*() const
  {
    return *this;
  }
  template <typename Difference>
  __host__ __device__ DetectWrongDifference operator+(Difference) const
  {
    return *this;
  }
  template <typename Index>
  __host__ __device__ DetectWrongDifference operator[](Index) const
  {
    return *this;
  }

  __device__ void operator=(long long difference) const
  {
    if (difference != 1)
    {
      *flag = false;
    }
  }
};

void TestAdjacentDifferenceWithBigIndexesHelper(int magnitude)
{
  const std::size_t elements = 1ll << magnitude;

  thrust::device_vector<bool> all_differences_correct(1, true);

  thrust::counting_iterator<long long> in(1);

  DetectWrongDifference out = {
    thrust::raw_pointer_cast(all_differences_correct.data())
  };

  AdjacentDifferenceCopy<READ_LEFT>(in, out, cub::Difference{}, elements);
  AssertEquals(all_differences_correct.front(), true);
}


void TestAdjacentDifferenceWithBigIndexes()
{
  TestAdjacentDifferenceWithBigIndexesHelper(30);
  TestAdjacentDifferenceWithBigIndexesHelper(31);
  TestAdjacentDifferenceWithBigIndexesHelper(32);
  TestAdjacentDifferenceWithBigIndexesHelper(33);
}

struct InvocationsCounter
{
  int *m_d_counts{};

  explicit InvocationsCounter(int *d_counts) : m_d_counts(d_counts) {}

  __device__ int operator()(int l, int /* r */) const 
  {
    atomicAdd(m_d_counts + l, 1);
    return l;
  }
};

void TestAdjacentDifferenceOpInvocationsNum(int num_items)
{
  auto in = thrust::make_counting_iterator(0);
  auto out = thrust::make_discard_iterator();

  thrust::device_vector<int> num_of_invocations(num_items, 0);
  InvocationsCounter op{thrust::raw_pointer_cast(num_of_invocations.data())};

  AdjacentDifferenceCopy<READ_LEFT>(in, out, op, num_items);
  AssertEquals(
    num_items - 1,
    thrust::count(num_of_invocations.begin() + 1, num_of_invocations.end(), 1));
  AssertEquals(0, num_of_invocations[0]);

  thrust::fill_n(num_of_invocations.begin(), num_items, 0);
  AdjacentDifferenceCopy<READ_RIGHT>(in, out, op, num_items);
  AssertEquals(
    num_items - 1,
    thrust::count(num_of_invocations.begin(), num_of_invocations.end() - 1, 1));
  AssertEquals(0, num_of_invocations[num_items - 1]);
}

void TestAdjacentDifferenceOpInvocationsNum()
{
  for (int num_items = 1; num_items < 4096; num_items *= 2)
  {
    TestAdjacentDifferenceOpInvocationsNum(num_items);
  }
}

int main(int argc, char** argv)
{
  CommandLineArgs args(argc, argv);

  // Initialize device
  CubDebugExit(args.DeviceInit());

  TestSize(0);
  for (std::size_t power_of_two = 2; power_of_two < 20; power_of_two += 2)
  {
    TestSize(1ull << power_of_two);
  }
  TestAdjacentDifferenceWithBigIndexes();
  TestAdjacentDifferenceOpInvocationsNum();

  return 0;
}
