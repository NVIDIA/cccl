// This example demonstrates a single-pass run-length decoding algorithm using thrust::inclusive_scan.
// The algorithm processes run-length encoded data and expands it in a single pass by computing
// offsets during the scan and filling the output array as the scan proceeds. The size of the output
// has to be known ahead of time.

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>

#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/__algorithm/equal.h>
#include <cuda/std/__algorithm/fill.h>
#include <cuda/std/cstdint>
#include <cuda/std/ranges>

#include <algorithm>
#include <iostream>

template <typename ValueType, typename CountType>
struct run
{
  ValueType value{};
  CountType count{0};
  CountType offset{0}; // Offset where we should begin outputting the run.
  CountType run_id{1}; // 1-index of the run in the input sequence; subtract by 1 to get the 0-index.

  __host__ __device__ friend run operator+(run l, run r)
  {
    return run{r.value, r.count, l.offset + l.count + r.offset, l.run_id + r.run_id};
  }
};

template <typename ValueType, typename CountType>
__host__ __device__ run(ValueType, CountType) -> run<ValueType, CountType>;

template <typename OutputIterator, typename CountType, typename ExpandedSizeIterator>
struct expand
{
  OutputIterator out;
  CountType out_size;
  CountType runs_size;
  ExpandedSizeIterator expanded_size;

  template <typename ValueType>
  __host__ __device__ CountType operator()(run<ValueType, CountType> r) const
  {
    cuda::std::size_t end = cuda::std::min(r.offset + r.count, out_size);
    cuda::std::fill(out + r.offset, out + end, r.value);
    if (r.run_id == runs_size) // If we're the last run, write the expanded size.
    {
      *expanded_size = end;
    }
    return end;
  }
};

template <typename OutputIterator, typename CountType, typename ExpandedSizeIterator>
__host__ __device__ expand(OutputIterator, CountType, CountType, ExpandedSizeIterator)
  -> expand<OutputIterator, CountType, ExpandedSizeIterator>;

template <typename ValueIterator, typename CountIterator, typename OutputIterator, typename CountType>
OutputIterator run_length_decode(
  ValueIterator values, CountIterator counts, CountType runs_size, OutputIterator out, CountType out_size)
{
  using ValueType = typename ValueIterator::value_type;

  // Zip the values and counts together and convert the resulting tuples to a named struct.
  auto runs = cuda::zip_transform_iterator(
    [] __host__ __device__(ValueType v, CountType c) {
      return run{v, c};
    },
    values,
    counts);

  // Allocate storage for the expanded size. If we were using CUB directly, we could read this
  // from the final ScanTileState.
  thrust::device_vector<CountType> expanded_size(1);

  // Iterator that writes the expanded sequence to the output and discards the actual scan results.
  auto expand_out = thrust::make_transform_output_iterator(
    thrust::make_discard_iterator(), expand{out, out_size, runs_size, expanded_size.begin()});

  // Scan to compute output offsets and then write the expanded sequence.
  thrust::inclusive_scan(thrust::device, runs, runs + runs_size, expand_out);

  // Calculate the end of the output sequence.
  return out + expanded_size[0];
}

int main()
{
  using ValueType = cuda::std::int32_t;
  using CountType = cuda::std::size_t;

  CountType size   = 8192;
  CountType repeat = 4;

  thrust::device_vector<ValueType> values(size);
  thrust::sequence(thrust::device, values.begin(), values.end(), 0);

  thrust::device_vector<CountType> counts(size);
  thrust::fill(thrust::device, counts.begin(), counts.end(), repeat);

  // Print first 32 elements of compressed input
  thrust::host_vector<ValueType> values_h(32);
  thrust::host_vector<CountType> counts_h(32);
  thrust::copy(values.begin(), values.begin() + 32, values_h.begin());
  thrust::copy(counts.begin(), counts.begin() + 32, counts_h.begin());

  std::cout << "Compressed input (first 32 runs):" << std::endl;
  for (CountType i = 0; i < 32; ++i)
  {
    std::cout << "(" << values_h[i] << "," << counts_h[i] << ") ";
  }
  std::cout << std::endl << std::endl;

  thrust::device_vector<ValueType> output(size * repeat);

  auto out_end = run_length_decode(values.begin(), counts.begin(), values.size(), output.begin(), output.size());

  if (CountType(out_end - output.begin()) != size * repeat)
  {
    std::cerr << "Error: Output size mismatch. Expected " << size * repeat << " but got "
              << static_cast<CountType>(out_end - output.begin()) << std::endl;
    return 1;
  }

  thrust::host_vector<ValueType> observed(size * repeat);
  thrust::copy(output.begin(), out_end, observed.begin());

  // Print first 32 elements of decompressed output
  std::cout << "Decompressed output (first 32 elements):" << std::endl;
  for (CountType i = 0; i < 32; ++i)
  {
    std::cout << observed[i] << " ";
  }
  std::cout << std::endl;

  auto gold = cuda::std::views::iota(CountType{0}, size * repeat) | cuda::std::views::transform([=](CountType idx) {
                return ValueType(idx / repeat);
              });

  if (!cuda::std::equal(observed.begin(), observed.end(), gold.begin()))
  {
    std::cerr << "Error: Output does not match expected values" << std::endl;
    return 1;
  }
}
