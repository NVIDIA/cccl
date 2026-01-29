#include <thrust/detail/config.h>

#include <thrust/gather.h>
#include <thrust/iterator/shuffle_iterator.h>
#include <thrust/random.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>
#include <thrust/sort.h>

#include <algorithm>
#include <limits>
#include <map>

#include <unittest/unittest.h>

struct iterator_shuffle_copy
{
  template <typename Iterator, typename ResultIterator>
  void operator()(Iterator first, Iterator last, ResultIterator result, thrust::default_random_engine& g)
  {
    auto shuffle_iter = thrust::make_shuffle_iterator(static_cast<uint64_t>(last - first), g);
    thrust::gather(shuffle_iter, shuffle_iter + (last - first), first, result);
  }
};

struct iterator_shuffle
{
  template <typename Iterator>
  void operator()(Iterator first, Iterator last, thrust::default_random_engine& g)
  {
    using thrust::system::detail::generic::select_system;
    using InputType = typename thrust::detail::it_value_t<Iterator>;
    using System    = typename thrust::iterator_system<Iterator>::type;
    System system;
    auto policy = select_system(system);
    thrust::detail::temporary_array<InputType, System> temp(policy, first, last);
    iterator_shuffle_copy{}(temp.begin(), temp.end(), first, g);
  }
};

struct thrust_shuffle
{
  template <typename Iterator>
  void operator()(Iterator first, Iterator last, thrust::default_random_engine& g)
  {
    thrust::shuffle(first, last, g);
  }
};

struct thrust_shuffle_copy
{
  template <typename Iterator>
  void operator()(Iterator first, Iterator last, Iterator result, thrust::default_random_engine& g)
  {
    thrust::shuffle_copy(first, last, result, g);
  }
};

template <class ShuffleFunc, typename Vector>
void TestShuffleSimpleBase()
{
  Vector data{0, 1, 2, 3, 4};
  Vector shuffled(data.begin(), data.end());
  thrust::default_random_engine g(2);
  ShuffleFunc{}(shuffled.begin(), shuffled.end(), g);
  thrust::sort(shuffled.begin(), shuffled.end());
  // Check all of our data is present
  // This only tests for strange conditions like duplicated elements
  ASSERT_EQUAL(shuffled, data);
}
template <typename Vector>
void TestShuffleSimple()
{
  TestShuffleSimpleBase<thrust_shuffle, Vector>();
}
template <typename Vector>
void TestShuffleSimpleIterator()
{
  TestShuffleSimpleBase<iterator_shuffle, Vector>();
}
DECLARE_VECTOR_UNITTEST(TestShuffleSimple);
DECLARE_VECTOR_UNITTEST(TestShuffleSimpleIterator);

template <typename ShuffleFunc, typename ShuffleCopyFunc, typename Vector>
void TestShuffleCopySimpleBase()
{
  Vector data{0, 1, 2, 3, 4};
  Vector shuffled(5);
  thrust::default_random_engine g(2);
  ShuffleCopyFunc{}(data.begin(), data.end(), shuffled.begin(), g);
  g.seed(2);
  ShuffleFunc{}(data.begin(), data.end(), g);
  ASSERT_EQUAL(shuffled, data);
}
template <typename Vector>
void TestShuffleCopySimple()
{
  TestShuffleCopySimpleBase<thrust_shuffle, thrust_shuffle_copy, Vector>();
}
template <typename Vector>
void TestShuffleCopySimpleIterator()
{
  TestShuffleCopySimpleBase<iterator_shuffle, iterator_shuffle_copy, Vector>();
}
DECLARE_VECTOR_UNITTEST(TestShuffleCopySimple);
DECLARE_VECTOR_UNITTEST(TestShuffleCopySimpleIterator);

template <typename ShuffleFunc, typename T>
void TestHostDeviceIdenticalBase(size_t m)
{
  thrust::host_vector<T> host_result(m);
  thrust::device_vector<T> device_result(m);
  thrust::sequence(host_result.begin(), host_result.end(), T{});
  thrust::sequence(device_result.begin(), device_result.end(), T{});

  thrust::default_random_engine host_g(183);
  thrust::default_random_engine device_g(183);

  ShuffleFunc{}(host_result.begin(), host_result.end(), host_g);
  ShuffleFunc{}(device_result.begin(), device_result.end(), device_g);

  ASSERT_EQUAL(device_result, host_result);
}
template <typename T>
void TestHostDeviceIdentical(size_t m)
{
  TestHostDeviceIdenticalBase<thrust_shuffle, T>(m);
}
template <typename T>
void TestHostDeviceIdenticalIterator(size_t m)
{
  TestHostDeviceIdenticalBase<iterator_shuffle, T>(m);
}
DECLARE_VARIABLE_UNITTEST(TestHostDeviceIdentical);
DECLARE_VARIABLE_UNITTEST(TestHostDeviceIdenticalIterator);

template <typename BijectionFunc, typename T>
void TestFunctionIsBijectionBase(size_t m)
{
  thrust::default_random_engine device_g(0xD5);
  BijectionFunc device_f(m, device_g);

  const size_t total_length = device_f.size();
  if (static_cast<double>(total_length) >= static_cast<double>(std::numeric_limits<T>::max()) || m == 0)
  {
    return;
  }
  ASSERT_LEQUAL(total_length, (std::max) (m * 2, size_t(256))); // Check the rounded up size is at most double the input

  auto device_result_it = thrust::make_transform_iterator(thrust::make_counting_iterator(T(0)), device_f);

  thrust::device_vector<T> unpermuted(total_length, T(0));

  // Run a scatter, this should copy each value to the index matching is value, the result should be in ascending order
  thrust::scatter(device_result_it,
                  device_result_it + static_cast<T>(total_length), // total_length is guaranteed to fit T
                  device_result_it,
                  unpermuted.begin());

  // Check every index is in the result, if any are missing then the function was not a bijection over [0,m)
  ASSERT_EQUAL(true, thrust::equal(unpermuted.begin(), unpermuted.end(), thrust::make_counting_iterator(T(0))));
}
template <typename T>
void TestFunctionIsBijection(size_t m)
{
  TestFunctionIsBijectionBase<thrust::detail::feistel_bijection, T>(m);
}
template <typename T>
void TestFunctionIsBijectionIterator(size_t m)
{
  TestFunctionIsBijectionBase<thrust::detail::random_bijection<uint64_t>, T>(m);
}
DECLARE_INTEGRAL_VARIABLE_UNITTEST(TestFunctionIsBijection);
DECLARE_INTEGRAL_VARIABLE_UNITTEST(TestFunctionIsBijectionIterator);

void TestFeistelBijectionLength()
{
  thrust::default_random_engine g(0xD5);

  uint64_t m = 345;
  thrust::detail::feistel_bijection f(m, g);
  ASSERT_EQUAL(f.size(), uint64_t(512));

  m = 256;
  f = thrust::detail::feistel_bijection(m, g);
  ASSERT_EQUAL(f.size(), uint64_t(256));

  m = 1;
  f = thrust::detail::feistel_bijection(m, g);
  ASSERT_EQUAL(f.size(), uint64_t(256));
}
DECLARE_UNITTEST(TestFeistelBijectionLength);

void TestShuffleIteratorConstructibleFromBijection()
{
  thrust::default_random_engine g(0xD5);

  thrust::detail::feistel_bijection f(32, g);
  thrust::shuffle_iterator<uint64_t, decltype(f)> it(f);

  g.seed(0xD5);
  thrust::detail::random_bijection<uint64_t> f2(f.size(), g);
  thrust::shuffle_iterator<uint64_t, decltype(f2)> it2(f2);

  g.seed(0xD5);
  thrust::shuffle_iterator<uint64_t> it3(f.size(), g);

  ASSERT_EQUAL(true, thrust::equal(thrust::device, it, it + f.size(), it2));
  ASSERT_EQUAL(true, thrust::equal(thrust::device, it, it + f.size(), it3));
}
DECLARE_UNITTEST(TestShuffleIteratorConstructibleFromBijection);

void TestShuffleAndPermutationIterator()
{
  thrust::default_random_engine g(0xD5);

  auto it = thrust::make_shuffle_iterator(32, g);

  thrust::device_vector<uint64_t> data(32);
  thrust::sequence(data.begin(), data.end(), 0);

  auto permute_it = thrust::make_permutation_iterator(data.begin(), it);

  thrust::device_vector<uint64_t> premute_vec(32);
  thrust::gather(it, it + 32, data.begin(), premute_vec.begin());

  ASSERT_EQUAL(true, thrust::equal(permute_it, permute_it + 32, premute_vec.begin()));
}
DECLARE_UNITTEST(TestShuffleAndPermutationIterator);

void TestShuffleIteratorStateless()
{
  thrust::default_random_engine g(0xD5);

  auto it = thrust::make_shuffle_iterator(32, g);

  ASSERT_EQUAL(*it, *it);
  ASSERT_EQUAL(*(it + 1), *(it + 1));
  ++it;
  ASSERT_EQUAL(*(it - 1), *(it - 1));
}
DECLARE_UNITTEST(TestShuffleIteratorStateless);

double inverse_erf(double x)
{
  double tt1, tt2, lnx, sgn;
  sgn = (x < 0) ? -1.0 : 1.0;

  x   = (1 - x) * (1 + x);
  lnx = cuda::std::log(x);

  tt1 = 2 / (3.14159265358979323846 * 0.147) + 0.5f * lnx;
  tt2 = 1 / (0.147) * lnx;

  return (sgn * cuda::std::sqrt(-tt1 + cuda::std::sqrt(tt1 * tt1 - tt2)));
}

// Individual input keys should be permuted to output locations with uniform
// probability. Perform chi-squared test with confidence 95%.
template <typename ShuffleFunc, typename Vector>
void TestShuffleKeyPositionBase()
{
  using T               = typename Vector::value_type;
  const int num_samples = 1000;
  const int n           = 20;
  thrust::default_random_engine g(0xD5);
  thrust::host_vector<double> expected_value(n);
  thrust::host_vector<T> sequence(n);
  thrust::sequence(sequence.begin(), sequence.end(), T(0));
  for (size_t i = 0; i < num_samples; ++i)
  {
    Vector shuffled(sequence.begin(), sequence.end());
    ShuffleFunc{}(shuffled.begin(), shuffled.end(), g);
    thrust::host_vector<T> tmp(shuffled.begin(), shuffled.end());
    for (size_t j = 0; j < n; ++j)
    {
      expected_value[j] += static_cast<double>(tmp[j]);
    }
  }

  double mu    = (n - 1) / 2.0;
  double sigma = cuda::std::sqrt((n * n - 1) / 12.0);
  double zmax  = 0.0;
  for (size_t i = 0; i < n; ++i)
  {
    double mean = expected_value[i] / double(num_samples);
    double z    = cuda::std::abs(mean - mu) / (sigma / cuda::std::sqrt(double(num_samples)));
    zmax        = cuda::std::max(zmax, cuda::std::abs(z));
  }

  double alpha = 0.05;
  double zcrit = inverse_erf(1.0 - alpha / (2.0 * n)) * cuda::std::sqrt(2.0);
  ASSERT_LESS(zmax, zcrit);
}
template <typename Vector>
void TestShuffleKeyPosition()
{
  TestShuffleKeyPositionBase<thrust_shuffle, Vector>();
}
template <typename Vector>
void TestShuffleKeyPositionIterator()
{
  TestShuffleKeyPositionBase<iterator_shuffle, Vector>();
}
DECLARE_INTEGRAL_VECTOR_UNITTEST(TestShuffleKeyPosition);
DECLARE_INTEGRAL_VECTOR_UNITTEST(TestShuffleKeyPositionIterator);

struct vector_compare
{
  template <typename VectorT>
  bool operator()(const VectorT& a, const VectorT& b) const
  {
    for (auto i = 0ull; i < a.size(); i++)
    {
      if (a[i] < b[i])
      {
        return true;
      }
      if (a[i] > b[i])
      {
        return false;
      }
    }
    return false;
  }
};

// Brute force check permutations are uniformly distributed on small input
// Uses a chi-squared test indicating 99% confidence the output is uniformly
// random
template <typename ShuffleFunc, typename Vector>
void TestShuffleUniformPermutationBase()
{
  using T = typename Vector::value_type;

  size_t m                  = 5;
  size_t num_samples        = 1000;
  size_t total_permutations = 1 * 2 * 3 * 4 * 5;
  std::map<thrust::host_vector<T>, size_t, vector_compare> permutation_counts;
  Vector sequence(m);
  thrust::sequence(sequence.begin(), sequence.end(), T(0));
  thrust::default_random_engine g(0xD5);
  for (auto i = 0ull; i < num_samples; i++)
  {
    ShuffleFunc{}(sequence.begin(), sequence.end(), g);
    thrust::host_vector<T> tmp(sequence.begin(), sequence.end());
    permutation_counts[tmp]++;
  }

  ASSERT_EQUAL(permutation_counts.size(), total_permutations);

  double chi_squared    = 0.0;
  double expected_count = static_cast<double>(num_samples) / total_permutations;
  for (auto kv : permutation_counts)
  {
    chi_squared += std::pow(expected_count - kv.second, 2) / expected_count;
  }
  // 119 degrees of freedom, 95% confidence
  const double critical_value = 145.461;
  ASSERT_LESS(chi_squared, critical_value);
}
template <typename Vector>
void TestShuffleUniformPermutation()
{
  TestShuffleUniformPermutationBase<thrust_shuffle, Vector>();
}
template <typename Vector>
void TestShuffleUniformPermutationIterator()
{
  TestShuffleUniformPermutationBase<iterator_shuffle, Vector>();
}
DECLARE_VECTOR_UNITTEST(TestShuffleUniformPermutation);
DECLARE_VECTOR_UNITTEST(TestShuffleUniformPermutationIterator);
