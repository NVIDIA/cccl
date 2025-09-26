// nvcc -arch=native -std=c++20 --extended-lambda --expt-relaxed-constexpr run_length_decode.cu -o run_length_decode

#include <thrust/fill.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>

#include <cuda/std/cstdint>
#include <cuda/functional>

#include <iostream>
#include <iterator>

template <typename ValueType, typename CountType>
struct run {
  ValueType value;
  CountType count;
  CountType offset;

  __host__ __device__ run()
    : value(ValueType{}), count(0), offset(0) {}
  __host__ __device__ run(ValueType value, CountType count)
    : value(value), count(count), offset(0) {}
  __host__ __device__ run(ValueType value, CountType count, CountType offset)
    : value(value), count(count), offset(offset) {}
  run(run const& other) = default;
  run& operator=(run const& other) = default;

  __host__ __device__ friend run operator+(run l, run r) {
    return run{r.value, r.count, l.offset + l.count + r.offset};
  }
};

template <typename OutputIterator, typename ValueType, typename CountType>
struct expand {
  OutputIterator out;
  cuda::std::size_t out_size;

  __host__ __device__
  expand(OutputIterator out, cuda::std::size_t out_size)
    : out(out), out_size(out_size) {}

  __host__ __device__
  CountType operator()(run<ValueType, CountType> r) const {
    printf("expanding %d into out[%llu:%llu]\n", r.value, r.offset, r.offset + r.count);
    thrust::fill(thrust::seq,
      out + r.offset, out + cuda::minimum()(r.offset + r.count, out_size), r.value);
    return r.offset + r.count;
  }
};

template <typename ValueIterator, typename CountIterator, typename OutputIterator>
void run_length_decode(
  ValueIterator values,
  CountIterator counts,
  cuda::std::size_t runs_size,
  OutputIterator out,
  cuda::std::size_t out_size)
{
  using ValueT = typename ValueIterator::value_type;
  using CountT = typename CountIterator::value_type;
  using RunT = run<ValueT, CountT>;
  auto runs = thrust::make_transform_iterator(
    thrust::make_zip_iterator(values, counts),
    [] __host__ __device__ (thrust::tuple<ValueT, CountT> tup) {
      return run{thrust::get<0>(tup), thrust::get<1>(tup)};
    });
  auto expand_out = thrust::make_transform_output_iterator(
    thrust::make_discard_iterator(),
    expand<OutputIterator, ValueT, CountT>(out, out_size));
  thrust::inclusive_scan(thrust::device, runs, runs + runs_size, expand_out);
}

int main() {
  cuda::std::size_t size = 32;
  cuda::std::size_t repeat = 4;

  thrust::device_vector<cuda::std::int32_t> values(size);
  thrust::sequence(thrust::device, values.begin(), values.end(), 0);

  thrust::device_vector<cuda::std::size_t> counts(size);
  thrust::fill(thrust::device, counts.begin(), counts.end(), repeat);

  thrust::device_vector<cuda::std::int32_t> output(size * repeat);

  run_length_decode(values.begin(), counts.begin(), values.size(), output.begin(), output.size());

  thrust::copy(output.begin(), output.end(), std::ostream_iterator<cuda::std::int32_t>(std::cout, "\n"));
}
