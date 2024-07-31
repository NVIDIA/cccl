#include <thrust/detail/alignment.h>

#include <unittest/unittest.h>

struct alignof_mock_0
{
  char a;
  char b;
}; // size: 2 * sizeof(char), alignment: sizeof(char)

struct alignof_mock_1
{
  int n;
  char c;
  // sizeof(int) - sizeof(char) bytes of padding
}; // size: 2 * sizeof(int), alignment: sizeof(int)

struct alignof_mock_2
{
  int n;
  char c;
  // sizeof(int) - sizeof(char) bytes of padding
}; // size: 2 * sizeof(int), alignment: sizeof(int)

struct alignof_mock_3
{
  char c;
  // sizeof(int) - sizeof(char) bytes of padding
  int n;
}; // size: 2 * sizeof(int), alignment: sizeof(int)

struct alignof_mock_4
{
  char c0;
  // sizeof(int) - sizeof(char) bytes of padding
  int n;
  char c1;
  // sizeof(int) - sizeof(char) bytes of padding
}; // size: 3 * sizeof(int), alignment: sizeof(int)

struct alignof_mock_5
{
  char c0;
  char c1;
  // sizeof(int) - 2 * sizeof(char) bytes of padding
  int n;
}; // size: 2 * sizeof(int), alignment: sizeof(int)

struct alignof_mock_6
{
  int n;
  char c0;
  char c1;
  // sizeof(int) - 2 * sizeof(char) bytes of padding
}; // size: 2 * sizeof(int), alignment: sizeof(int)

void test_alignof_mocks_sizes()
{
  ASSERT_EQUAL(sizeof(alignof_mock_0), 2 * sizeof(char));
  ASSERT_EQUAL(sizeof(alignof_mock_1), 2 * sizeof(int));
  ASSERT_EQUAL(sizeof(alignof_mock_2), 2 * sizeof(int));
  ASSERT_EQUAL(sizeof(alignof_mock_3), 2 * sizeof(int));
  ASSERT_EQUAL(sizeof(alignof_mock_4), 3 * sizeof(int));
  ASSERT_EQUAL(sizeof(alignof_mock_5), 2 * sizeof(int));
  ASSERT_EQUAL(sizeof(alignof_mock_6), 2 * sizeof(int));
}
DECLARE_UNITTEST(test_alignof_mocks_sizes);

void test_alignof()
{
  ASSERT_EQUAL(alignof(bool), sizeof(bool));
  ASSERT_EQUAL(alignof(signed char), sizeof(signed char));
  ASSERT_EQUAL(alignof(unsigned char), sizeof(unsigned char));
  ASSERT_EQUAL(alignof(char), sizeof(char));
  ASSERT_EQUAL(alignof(short int), sizeof(short int));
  ASSERT_EQUAL(alignof(unsigned short int), sizeof(unsigned short int));
  ASSERT_EQUAL(alignof(int), sizeof(int));
  ASSERT_EQUAL(alignof(unsigned int), sizeof(unsigned int));
  ASSERT_EQUAL(alignof(long int), sizeof(long int));
  ASSERT_EQUAL(alignof(unsigned long int), sizeof(unsigned long int));
  ASSERT_EQUAL(alignof(long long int), sizeof(long long int));
  ASSERT_EQUAL(alignof(unsigned long long int), sizeof(unsigned long long int));
  ASSERT_EQUAL(alignof(float), sizeof(float));
  ASSERT_EQUAL(alignof(double), sizeof(double));
  ASSERT_EQUAL(alignof(long double), sizeof(long double));

  ASSERT_EQUAL(alignof(alignof_mock_0), sizeof(char));
  ASSERT_EQUAL(alignof(alignof_mock_1), sizeof(int));
  ASSERT_EQUAL(alignof(alignof_mock_2), sizeof(int));
  ASSERT_EQUAL(alignof(alignof_mock_3), sizeof(int));
  ASSERT_EQUAL(alignof(alignof_mock_4), sizeof(int));
  ASSERT_EQUAL(alignof(alignof_mock_5), sizeof(int));
  ASSERT_EQUAL(alignof(alignof_mock_6), sizeof(int));
}
DECLARE_UNITTEST(test_alignof);

void test_alignment_of()
{
  ASSERT_EQUAL(thrust::detail::alignment_of<bool>::value, sizeof(bool));
  ASSERT_EQUAL(thrust::detail::alignment_of<signed char>::value, sizeof(signed char));
  ASSERT_EQUAL(thrust::detail::alignment_of<unsigned char>::value, sizeof(unsigned char));
  ASSERT_EQUAL(thrust::detail::alignment_of<char>::value, sizeof(char));
  ASSERT_EQUAL(thrust::detail::alignment_of<short int>::value, sizeof(short int));
  ASSERT_EQUAL(thrust::detail::alignment_of<unsigned short int>::value, sizeof(unsigned short int));
  ASSERT_EQUAL(thrust::detail::alignment_of<int>::value, sizeof(int));
  ASSERT_EQUAL(thrust::detail::alignment_of<unsigned int>::value, sizeof(unsigned int));
  ASSERT_EQUAL(thrust::detail::alignment_of<long int>::value, sizeof(long int));
  ASSERT_EQUAL(thrust::detail::alignment_of<unsigned long int>::value, sizeof(unsigned long int));
  ASSERT_EQUAL(thrust::detail::alignment_of<long long int>::value, sizeof(long long int));
  ASSERT_EQUAL(thrust::detail::alignment_of<unsigned long long int>::value, sizeof(unsigned long long int));
  ASSERT_EQUAL(thrust::detail::alignment_of<float>::value, sizeof(float));
  ASSERT_EQUAL(thrust::detail::alignment_of<double>::value, sizeof(double));
  ASSERT_EQUAL(thrust::detail::alignment_of<long double>::value, sizeof(long double));

  ASSERT_EQUAL(thrust::detail::alignment_of<alignof_mock_0>::value, sizeof(char));
  ASSERT_EQUAL(thrust::detail::alignment_of<alignof_mock_1>::value, sizeof(int));
  ASSERT_EQUAL(thrust::detail::alignment_of<alignof_mock_2>::value, sizeof(int));
  ASSERT_EQUAL(thrust::detail::alignment_of<alignof_mock_3>::value, sizeof(int));
  ASSERT_EQUAL(thrust::detail::alignment_of<alignof_mock_4>::value, sizeof(int));
  ASSERT_EQUAL(thrust::detail::alignment_of<alignof_mock_5>::value, sizeof(int));
  ASSERT_EQUAL(thrust::detail::alignment_of<alignof_mock_6>::value, sizeof(int));
}
DECLARE_UNITTEST(test_alignment_of);

template <std::size_t Align>
void test_aligned_type_instantiation()
{
  using type = typename thrust::detail::aligned_type<Align>::type;
  ASSERT_GEQUAL(sizeof(type), 1lu);
  ASSERT_EQUAL(alignof(type), Align);
  ASSERT_EQUAL(thrust::detail::alignment_of<type>::value, Align);
}

void test_aligned_type()
{
  test_aligned_type_instantiation<1>();
  test_aligned_type_instantiation<2>();
  test_aligned_type_instantiation<4>();
  test_aligned_type_instantiation<8>();
  test_aligned_type_instantiation<16>();
  test_aligned_type_instantiation<32>();
  test_aligned_type_instantiation<64>();
  test_aligned_type_instantiation<128>();
}
DECLARE_UNITTEST(test_aligned_type);

void test_max_align_t()
{
  ASSERT_GEQUAL(alignof(thrust::detail::max_align_t), alignof(bool));
  ASSERT_GEQUAL(alignof(thrust::detail::max_align_t), alignof(signed char));
  ASSERT_GEQUAL(alignof(thrust::detail::max_align_t), alignof(unsigned char));
  ASSERT_GEQUAL(alignof(thrust::detail::max_align_t), alignof(char));
  ASSERT_GEQUAL(alignof(thrust::detail::max_align_t), alignof(short int));
  ASSERT_GEQUAL(alignof(thrust::detail::max_align_t), alignof(unsigned short int));
  ASSERT_GEQUAL(alignof(thrust::detail::max_align_t), alignof(int));
  ASSERT_GEQUAL(alignof(thrust::detail::max_align_t), alignof(unsigned int));
  ASSERT_GEQUAL(alignof(thrust::detail::max_align_t), alignof(long int));
  ASSERT_GEQUAL(alignof(thrust::detail::max_align_t), alignof(unsigned long int));
  ASSERT_GEQUAL(alignof(thrust::detail::max_align_t), alignof(long long int));
  ASSERT_GEQUAL(alignof(thrust::detail::max_align_t), alignof(unsigned long long int));
  ASSERT_GEQUAL(alignof(thrust::detail::max_align_t), alignof(float));
  ASSERT_GEQUAL(alignof(thrust::detail::max_align_t), alignof(double));
  ASSERT_GEQUAL(alignof(thrust::detail::max_align_t), alignof(long double));
}
DECLARE_UNITTEST(test_max_align_t);

void test_aligned_reinterpret_cast()
{
  thrust::detail::aligned_type<1>* a1 = 0;

  thrust::detail::aligned_type<2>* a2 = 0;

  // Cast to type with stricter (larger) alignment requirement.
  a2 = thrust::detail::aligned_reinterpret_cast<thrust::detail::aligned_type<2>*>(a1);

  // Cast to type with less strict (smaller) alignment requirement.
  a1 = thrust::detail::aligned_reinterpret_cast<thrust::detail::aligned_type<1>*>(a2);
}
DECLARE_UNITTEST(test_aligned_reinterpret_cast);
