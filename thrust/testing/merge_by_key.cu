#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/retag.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/merge.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <unittest/unittest.h>

template <typename Vector>
void TestMergeByKeySimple()
{
  const Vector a_key{0, 2, 4}, a_val{13, 7, 42}, b_key{0, 3, 3, 4}, b_val{42, 42, 7, 13};
  Vector ref_key{0, 0, 2, 3, 3, 4, 4}, ref_val{13, 42, 7, 42, 7, 42, 13};

  Vector result_key(7), result_val(7);

  const auto ends = thrust::merge_by_key(
    a_key.begin(),
    a_key.end(),
    b_key.begin(),
    b_key.end(),
    a_val.begin(),
    b_val.begin(),
    result_key.begin(),
    result_val.begin());

  ASSERT_EQUAL_QUIET(result_key.end(), ends.first);
  ASSERT_EQUAL_QUIET(result_val.end(), ends.second);
  ASSERT_EQUAL(ref_key, result_key);
  ASSERT_EQUAL(ref_val, result_val);
}
DECLARE_VECTOR_UNITTEST(TestMergeByKeySimple);

template <typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename InputIterator4,
          typename OutputIterator1,
          typename OutputIterator2>
thrust::pair<OutputIterator1, OutputIterator2> merge_by_key(
  my_system& system,
  InputIterator1,
  InputIterator1,
  InputIterator2,
  InputIterator2,
  InputIterator3,
  InputIterator4,
  OutputIterator1 keys_result,
  OutputIterator2 values_result)
{
  system.validate_dispatch();
  return thrust::make_pair(keys_result, values_result);
}

void TestMergeByKeyDispatchExplicit()
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::merge_by_key(
    sys, vec.begin(), vec.begin(), vec.begin(), vec.begin(), vec.begin(), vec.begin(), vec.begin(), vec.begin());

  ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestMergeByKeyDispatchExplicit);

template <typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename InputIterator4,
          typename OutputIterator1,
          typename OutputIterator2>
thrust::pair<OutputIterator1, OutputIterator2> merge_by_key(
  my_tag,
  InputIterator1,
  InputIterator1,
  InputIterator2,
  InputIterator2,
  InputIterator3,
  InputIterator4,
  OutputIterator1 keys_result,
  OutputIterator2 values_result)
{
  *keys_result = 13;
  return thrust::make_pair(keys_result, values_result);
}

void TestMergeByKeyDispatchImplicit()
{
  thrust::device_vector<int> vec(1);

  thrust::merge_by_key(
    thrust::retag<my_tag>(vec.begin()),
    thrust::retag<my_tag>(vec.begin()),
    thrust::retag<my_tag>(vec.begin()),
    thrust::retag<my_tag>(vec.begin()),
    thrust::retag<my_tag>(vec.begin()),
    thrust::retag<my_tag>(vec.begin()),
    thrust::retag<my_tag>(vec.begin()),
    thrust::retag<my_tag>(vec.begin()));

  ASSERT_EQUAL(13, vec.front());
}

template <typename T, typename CompareOp, typename... Args>
auto call_merge_by_key(Args&&... args) -> decltype(thrust::merge_by_key(std::forward<Args>(args)...))
{
  _CCCL_IF_CONSTEXPR (::cuda::std::is_void<CompareOp>::value)
  {
    return thrust::merge_by_key(std::forward<Args>(args)...);
  }
  else
  {
    // TODO(bgruber): remove next line in C++17 and pass CompareOp{} directly to stable_sort
    using C = ::cuda::std::__conditional_t<::cuda::std::is_void<CompareOp>::value, thrust::less<T>, CompareOp>;
    return thrust::merge_by_key(std::forward<Args>(args)..., C{});
  }
  _CCCL_UNREACHABLE();
}

DECLARE_UNITTEST(TestMergeByKeyDispatchImplicit);

template <typename T, typename CompareOp = void>
void TestMergeByKey(size_t n)
{
  const auto random_keys = unittest::random_integers<unittest::int8_t>(n);
  const auto random_vals = unittest::random_integers<unittest::int8_t>(n);

  const size_t denominators[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  for (const auto& denom : denominators)
  {
    const size_t size_a = n / denom;

    thrust::host_vector<T> h_a_keys(random_keys.begin(), random_keys.begin() + size_a);
    thrust::host_vector<T> h_b_keys(random_keys.begin() + size_a, random_keys.end());

    const thrust::host_vector<T> h_a_vals(random_vals.begin(), random_vals.begin() + size_a);
    const thrust::host_vector<T> h_b_vals(random_vals.begin() + size_a, random_vals.end());

    _CCCL_IF_CONSTEXPR (::cuda::std::is_void<CompareOp>::value)
    {
      thrust::stable_sort(h_a_keys.begin(), h_a_keys.end());
      thrust::stable_sort(h_b_keys.begin(), h_b_keys.end());
    }
    else
    {
      // TODO(bgruber): remove next line in C++17 and pass CompareOp{} directly to stable_sort
      using C = ::cuda::std::__conditional_t<::cuda::std::is_void<CompareOp>::value, thrust::less<T>, CompareOp>;
      thrust::stable_sort(h_a_keys.begin(), h_a_keys.end(), C{});
      thrust::stable_sort(h_b_keys.begin(), h_b_keys.end(), C{});
    }

    const thrust::device_vector<T> d_a_keys = h_a_keys;
    const thrust::device_vector<T> d_b_keys = h_b_keys;

    const thrust::device_vector<T> d_a_vals = h_a_vals;
    const thrust::device_vector<T> d_b_vals = h_b_vals;

    thrust::host_vector<T> h_result_keys(n);
    thrust::host_vector<T> h_result_vals(n);

    thrust::device_vector<T> d_result_keys(n);
    thrust::device_vector<T> d_result_vals(n);

    const auto h_end = call_merge_by_key<T, CompareOp>(
      h_a_keys.begin(),
      h_a_keys.end(),
      h_b_keys.begin(),
      h_b_keys.end(),
      h_a_vals.begin(),
      h_b_vals.begin(),
      h_result_keys.begin(),
      h_result_vals.begin());

    h_result_keys.erase(h_end.first, h_result_keys.end());
    h_result_vals.erase(h_end.second, h_result_vals.end());

    const auto d_end = call_merge_by_key<T, CompareOp>(
      d_a_keys.begin(),
      d_a_keys.end(),
      d_b_keys.begin(),
      d_b_keys.end(),
      d_a_vals.begin(),
      d_b_vals.begin(),
      d_result_keys.begin(),
      d_result_vals.begin());
    d_result_keys.erase(d_end.first, d_result_keys.end());
    d_result_vals.erase(d_end.second, d_result_vals.end());

    ASSERT_EQUAL(h_result_keys, d_result_keys);
    ASSERT_EQUAL(h_result_vals, d_result_vals);
    ASSERT_EQUAL(true, h_end.first == h_result_keys.end());
    ASSERT_EQUAL(true, h_end.second == h_result_vals.end());
    ASSERT_EQUAL(true, d_end.first == d_result_keys.end());
    ASSERT_EQUAL(true, d_end.second == d_result_vals.end());
  }
}
DECLARE_VARIABLE_UNITTEST(TestMergeByKey);

template <typename T>
void TestMergeByKeyToDiscardIterator(size_t n)
{
  auto h_a_keys = unittest::random_integers<T>(n);
  auto h_b_keys = unittest::random_integers<T>(n);

  const auto h_a_vals = unittest::random_integers<T>(n);
  const auto h_b_vals = unittest::random_integers<T>(n);

  thrust::stable_sort(h_a_keys.begin(), h_a_keys.end());
  thrust::stable_sort(h_b_keys.begin(), h_b_keys.end());

  const thrust::device_vector<T> d_a_keys = h_a_keys;
  const thrust::device_vector<T> d_b_keys = h_b_keys;

  const thrust::device_vector<T> d_a_vals = h_a_vals;
  const thrust::device_vector<T> d_b_vals = h_b_vals;

  using discard_pair = thrust::pair<thrust::discard_iterator<>, thrust::discard_iterator<>>;

  const discard_pair h_result = thrust::merge_by_key(
    h_a_keys.begin(),
    h_a_keys.end(),
    h_b_keys.begin(),
    h_b_keys.end(),
    h_a_vals.begin(),
    h_b_vals.begin(),
    thrust::make_discard_iterator(),
    thrust::make_discard_iterator());

  const discard_pair d_result = thrust::merge_by_key(
    d_a_keys.begin(),
    d_a_keys.end(),
    d_b_keys.begin(),
    d_b_keys.end(),
    d_a_vals.begin(),
    d_b_vals.begin(),
    thrust::make_discard_iterator(),
    thrust::make_discard_iterator());

  const thrust::discard_iterator<> reference(2 * n);

  ASSERT_EQUAL_QUIET(reference, h_result.first);
  ASSERT_EQUAL_QUIET(reference, h_result.second);
  ASSERT_EQUAL_QUIET(reference, d_result.first);
  ASSERT_EQUAL_QUIET(reference, d_result.second);
}
DECLARE_VARIABLE_UNITTEST(TestMergeByKeyToDiscardIterator);

template <typename T>
void TestMergeByKeyDescending(size_t n)
{
  TestMergeByKey<T, thrust::greater<T>>(n);
}
DECLARE_VARIABLE_UNITTEST(TestMergeByKeyDescending);

struct def_level_fn
{
  _CCCL_DEVICE std::uint32_t operator()(int i) const
  {
    return static_cast<uint32_t>(i + 10);
  }
};

struct offset_transform
{
  _CCCL_DEVICE int operator()(int i) const
  {
    return i + 1;
  }
};

// Tests the use of thrust::merge_by_key similar to cuDF in
// https://github.com/rapidsai/cudf/blob/branch-24.08/cpp/src/lists/dremel.cu#L413
void TestMergeByKeyFromCuDFDremel()
{
  // TODO(bgruber): I have no idea what this code is actually computing, but I tried to replicate the types/iterators
  constexpr std::ptrdiff_t empties_size = 123;
  constexpr int max_vals_size           = 225;
  constexpr int level                   = 4;
  constexpr int curr_rep_values_size    = 0;

  thrust::device_vector<int> empties(empties_size, 42);
  thrust::device_vector<int> empties_idx(empties_size, 13);

  thrust::device_vector<std::uint8_t> temp_rep_vals(max_vals_size);
  thrust::device_vector<std::uint8_t> temp_def_vals(max_vals_size);
  thrust::device_vector<std::uint8_t> rep_level(max_vals_size);
  thrust::device_vector<std::uint8_t> def_level(max_vals_size);

  auto offset_transformer  = offset_transform{};
  auto transformed_empties = thrust::make_transform_iterator(empties.begin(), offset_transformer);

  auto input_parent_rep_it = thrust::make_constant_iterator(level);
  auto input_parent_def_it = thrust::make_transform_iterator(empties_idx.begin(), def_level_fn{});
  auto input_parent_zip_it = thrust::make_zip_iterator(input_parent_rep_it, input_parent_def_it);
  auto input_child_zip_it  = thrust::make_zip_iterator(temp_rep_vals.begin(), temp_def_vals.begin());
  auto output_zip_it       = thrust::make_zip_iterator(rep_level.begin(), def_level.begin());

  thrust::merge_by_key(
    transformed_empties,
    transformed_empties + empties_size,
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(curr_rep_values_size),
    input_parent_zip_it,
    input_child_zip_it,
    thrust::make_discard_iterator(),
    output_zip_it);

  thrust::device_vector<std::uint8_t> reference_rep_level(max_vals_size);
  thrust::fill(reference_rep_level.begin(), reference_rep_level.begin() + empties_size, level);

  thrust::device_vector<std::uint8_t> reference_def_level(max_vals_size);
  thrust::fill(reference_def_level.begin(), reference_def_level.begin() + empties_size, 13 + 10);

  ASSERT_EQUAL(reference_rep_level, rep_level);
  ASSERT_EQUAL(reference_def_level, def_level);
}
DECLARE_UNITTEST(TestMergeByKeyFromCuDFDremel);
