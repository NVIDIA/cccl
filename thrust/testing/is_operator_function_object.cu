#include <thrust/detail/static_assert.h>
#include <thrust/type_traits/is_operator_less_or_greater_function_object.h>
#include <thrust/type_traits/is_operator_plus_function_object.h>

#include <unittest/unittest.h>

static_assert(thrust::is_operator_less_function_object<std::less<>>::value);
static_assert(thrust::is_operator_greater_function_object<std::greater<>>::value);
static_assert(thrust::is_operator_less_or_greater_function_object<std::less<>>::value);
static_assert(thrust::is_operator_less_or_greater_function_object<std::greater<>>::value);
static_assert(thrust::is_operator_plus_function_object<std::plus<>>::value);

template <typename T>
_CCCL_HOST void test_is_operator_less_function_object()
{
  static_assert(thrust::is_operator_less_function_object<::cuda::std::less<T>>::value);
  static_assert(!thrust::is_operator_less_function_object<::cuda::std::greater<T>>::value);
  static_assert(!thrust::is_operator_less_function_object<::cuda::std::less_equal<T>>::value);
  static_assert(!thrust::is_operator_less_function_object<::cuda::std::greater_equal<T>>::value);
  static_assert(thrust::is_operator_less_function_object<std::less<T>>::value);
  static_assert(!thrust::is_operator_less_function_object<std::greater<T>>::value);
  static_assert(!thrust::is_operator_less_function_object<std::less_equal<T>>::value);
  static_assert(!thrust::is_operator_less_function_object<std::greater_equal<T>>::value);
  static_assert(!thrust::is_operator_less_function_object<T>::value);
}
DECLARE_GENERIC_UNITTEST(test_is_operator_less_function_object);

template <typename T>
_CCCL_HOST void test_is_operator_greater_function_object()
{
  static_assert(!thrust::is_operator_greater_function_object<::cuda::std::less<T>>::value);
  static_assert(thrust::is_operator_greater_function_object<::cuda::std::greater<T>>::value);
  static_assert(!thrust::is_operator_greater_function_object<::cuda::std::less_equal<T>>::value);
  static_assert(!thrust::is_operator_greater_function_object<::cuda::std::greater_equal<T>>::value);
  static_assert(!thrust::is_operator_greater_function_object<std::less<T>>::value);
  static_assert(thrust::is_operator_greater_function_object<std::greater<T>>::value);
  static_assert(!thrust::is_operator_greater_function_object<std::less_equal<T>>::value);
  static_assert(!thrust::is_operator_greater_function_object<std::greater_equal<T>>::value);
  static_assert(!thrust::is_operator_greater_function_object<T>::value);
}
DECLARE_GENERIC_UNITTEST(test_is_operator_greater_function_object);

template <typename T>
_CCCL_HOST void test_is_operator_less_or_greater_function_object()
{
  static_assert(thrust::is_operator_less_or_greater_function_object<::cuda::std::less<T>>::value);
  static_assert(thrust::is_operator_less_or_greater_function_object<::cuda::std::greater<T>>::value);
  static_assert(!thrust::is_operator_less_or_greater_function_object<::cuda::std::less_equal<T>>::value);
  static_assert(!thrust::is_operator_less_or_greater_function_object<::cuda::std::greater_equal<T>>::value);
  static_assert(thrust::is_operator_less_or_greater_function_object<std::less<T>>::value);
  static_assert(thrust::is_operator_less_or_greater_function_object<std::greater<T>>::value);
  static_assert(!thrust::is_operator_less_or_greater_function_object<std::less_equal<T>>::value);
  static_assert(!thrust::is_operator_less_or_greater_function_object<std::greater_equal<T>>::value);
  static_assert(!thrust::is_operator_less_or_greater_function_object<T>::value);
}
DECLARE_GENERIC_UNITTEST(test_is_operator_less_or_greater_function_object);

template <typename T>
_CCCL_HOST void test_is_operator_plus_function_object()
{
  static_assert(thrust::is_operator_plus_function_object<::cuda::std::plus<T>>::value);
  static_assert(!thrust::is_operator_plus_function_object<::cuda::std::minus<T>>::value);
  static_assert(!thrust::is_operator_plus_function_object<::cuda::std::less<T>>::value);
  static_assert(!thrust::is_operator_plus_function_object<::cuda::std::greater<T>>::value);
  static_assert(thrust::is_operator_plus_function_object<std::plus<T>>::value);
  static_assert(!thrust::is_operator_plus_function_object<std::minus<T>>::value);
  static_assert(!thrust::is_operator_plus_function_object<std::less<T>>::value);
  static_assert(!thrust::is_operator_plus_function_object<std::greater<T>>::value);
  static_assert(!thrust::is_operator_plus_function_object<T>::value);
}
DECLARE_GENERIC_UNITTEST(test_is_operator_plus_function_object);
