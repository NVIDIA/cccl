#include <thrust/detail/allocator/allocator_traits.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

#include <unittest/unittest.h>

static const size_t num_samples = 10000;

template <typename Vector, typename U>
struct rebind_vector;

template <typename T, typename U, typename Allocator>
struct rebind_vector<thrust::host_vector<T, Allocator>, U>
{
  using alloc_traits = typename thrust::detail::allocator_traits<Allocator>;
  using new_alloc    = typename alloc_traits::template rebind_alloc<U>;
  using type         = thrust::host_vector<U, new_alloc>;
};

template <typename T, typename U, typename Allocator>
struct rebind_vector<thrust::device_vector<T, Allocator>, U>
{
  using type = thrust::device_vector<U, typename Allocator::template rebind<U>::other>;
};

template <typename T, typename U, typename Allocator>
struct rebind_vector<thrust::universal_vector<T, Allocator>, U>
{
  using type = thrust::universal_vector<U, typename Allocator::template rebind<U>::other>;
};

#define BINARY_FUNCTIONAL_PLACEHOLDERS_TEST(name, reference_operator, functor)                        \
  template <typename Vector>                                                                          \
  void TestFunctionalPlaceholders##name()                                                             \
  {                                                                                                   \
    using T           = typename Vector::value_type;                                                  \
    using bool_vector = typename rebind_vector<Vector, bool>::type;                                   \
    Vector lhs        = unittest::random_samples<T>(num_samples);                                     \
    Vector rhs        = unittest::random_samples<T>(num_samples);                                     \
                                                                                                      \
    bool_vector reference(lhs.size());                                                                \
    thrust::transform(lhs.begin(), lhs.end(), rhs.begin(), reference.begin(), functor<T>());          \
                                                                                                      \
    using namespace thrust::placeholders;                                                             \
    bool_vector result(lhs.size());                                                                   \
    thrust::transform(lhs.begin(), lhs.end(), rhs.begin(), result.begin(), _1 reference_operator _2); \
                                                                                                      \
    ASSERT_EQUAL(reference, result);                                                                  \
  }                                                                                                   \
  DECLARE_VECTOR_UNITTEST(TestFunctionalPlaceholders##name);

BINARY_FUNCTIONAL_PLACEHOLDERS_TEST(LogicalAnd, &&, thrust::logical_and);
BINARY_FUNCTIONAL_PLACEHOLDERS_TEST(LogicalOr, ||, thrust::logical_or);

template <typename Vector>
void TestFunctionalPlaceholdersLogicalNot()
{
  using T           = typename Vector::value_type;
  using bool_vector = typename rebind_vector<Vector, bool>::type;
  Vector input      = unittest::random_samples<T>(num_samples);

  if (input.size() > 0)
  {
    // produce at least one true in the output
    input[0] = T(0);
  } // end if

  bool_vector reference(input.size());
  thrust::transform(input.begin(), input.end(), reference.begin(), thrust::logical_not<T>());

  using namespace thrust::placeholders;
  bool_vector result(input.size());
  thrust::transform(input.begin(), input.end(), result.begin(), !_1);

  ASSERT_EQUAL(reference, result);
}
DECLARE_INTEGRAL_VECTOR_UNITTEST(TestFunctionalPlaceholdersLogicalNot);
