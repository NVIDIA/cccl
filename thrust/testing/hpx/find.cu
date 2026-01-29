#include <thrust/find.h>

#include "test_executor.h"
#include <unittest/unittest.h>

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4244 4267) // possible loss of data

template <typename ExecutionPolicy>
void TestFind(ExecutionPolicy exec)
{
  using Vector = thrust::device_vector<int>;
  using T      = typename Vector::value_type;

  Vector input{0, 5, 3, 7, 9, 2, 1, 4, 6, 8};

  // Test find
  {
    typename Vector::iterator result = thrust::find(exec, input.begin(), input.end(), 7);
    ASSERT_EQUAL(result - input.begin(), 3);
  }

  // Test find with value not present
  {
    typename Vector::iterator result = thrust::find(exec, input.begin(), input.end(), 99);
    ASSERT_EQUAL(result - input.begin(), input.end() - input.begin());
  }

  // Test find_if with predicate
  {
    auto is_greater_than_5 = [](T x) { return x > 5; };
    typename Vector::iterator result = thrust::find_if(exec, input.begin(), input.end(), is_greater_than_5);
    ASSERT_EQUAL(*result, 7); // First element > 5
  }

  // Test find_if with no match
  {
    auto is_greater_than_10 = [](T x) { return x > 10; };
    typename Vector::iterator result = thrust::find_if(exec, input.begin(), input.end(), is_greater_than_10);
    ASSERT_EQUAL(result - input.begin(), input.end() - input.begin());
  }

  // Test find_if_not
  {
    auto is_even = [](T x) { return x % 2 == 0; };
    typename Vector::iterator result = thrust::find_if_not(exec, input.begin(), input.end(), is_even);
    ASSERT_EQUAL(*result, 5); // First odd number
  }

  // Test find_if_not with no match
  {
    auto is_single_digit = [](T x) { return x < 10; };
    typename Vector::iterator result = thrust::find_if_not(exec, input.begin(), input.end(), is_single_digit);
    ASSERT_EQUAL(result - input.begin(), input.end() - input.begin());
  }
}

template <typename ExecutionPolicy>
void TestFindWithDifferentTypes(ExecutionPolicy exec)
{
  // Test with counting iterator
  {
    thrust::counting_iterator<int> first(0);
    thrust::counting_iterator<int> last(10);
    
    auto result = thrust::find(exec, first, last, 5);
    ASSERT_EQUAL(*result, 5);
  }

  // Test with constant iterator
  {
    thrust::counting_iterator<int> first(42);
    thrust::counting_iterator<int> last = first + 10;
    
    auto result = thrust::find(exec, first, last, 42);
    ASSERT_EQUAL(result - first, 0);
  }
}

void TestFindPar()
{
  TestFind(thrust::hpx::par);
  TestFindWithDifferentTypes(thrust::hpx::par);
}
DECLARE_UNITTEST(TestFindPar);

void TestFindParOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestFind(thrust::hpx::par.on(exec));
  TestFindWithDifferentTypes(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestFindParOnSequencedExecutor);

void TestFindParOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestFind(thrust::hpx::par.on(exec));
  TestFindWithDifferentTypes(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestFindParOnTestSyncExecutor);

void TestFindParWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestFind(thrust::hpx::par.with(acs));
  TestFindWithDifferentTypes(thrust::hpx::par.with(acs));
}
DECLARE_UNITTEST(TestFindParWithAutoChunkSize);

void TestFindParUnseq()
{
  TestFind(thrust::hpx::par_unseq);
  TestFindWithDifferentTypes(thrust::hpx::par_unseq);
}
DECLARE_UNITTEST(TestFindParUnseq);

void TestFindParUnseqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestFind(thrust::hpx::par_unseq.on(exec));
  TestFindWithDifferentTypes(thrust::hpx::par_unseq.on(exec));
}
DECLARE_UNITTEST(TestFindParUnseqOnSequencedExecutor);

void TestFindParUnseqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestFind(thrust::hpx::par_unseq.on(exec));
  TestFindWithDifferentTypes(thrust::hpx::par_unseq.on(exec));
}
DECLARE_UNITTEST(TestFindParUnseqOnTestSyncExecutor);

void TestFindParUnseqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestFind(thrust::hpx::par_unseq.with(acs));
  TestFindWithDifferentTypes(thrust::hpx::par_unseq.with(acs));
}
DECLARE_UNITTEST(TestFindParUnseqWithAutoChunkSize);

void TestFindSeq()
{
  TestFind(thrust::hpx::seq);
  TestFindWithDifferentTypes(thrust::hpx::seq);
}
DECLARE_UNITTEST(TestFindSeq);

void TestFindSeqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestFind(thrust::hpx::seq.on(exec));
  TestFindWithDifferentTypes(thrust::hpx::seq.on(exec));
}
DECLARE_UNITTEST(TestFindSeqOnSequencedExecutor);

void TestFindSeqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestFind(thrust::hpx::seq.on(exec));
  TestFindWithDifferentTypes(thrust::hpx::seq.on(exec));
}
DECLARE_UNITTEST(TestFindSeqOnTestSyncExecutor);

void TestFindSeqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestFind(thrust::hpx::seq.with(acs));
  TestFindWithDifferentTypes(thrust::hpx::seq.with(acs));
}
DECLARE_UNITTEST(TestFindSeqWithAutoChunkSize);

void TestFindUnseq()
{
  TestFind(thrust::hpx::unseq);
  TestFindWithDifferentTypes(thrust::hpx::unseq);
}
DECLARE_UNITTEST(TestFindUnseq);

void TestFindUnseqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestFind(thrust::hpx::unseq.on(exec));
  TestFindWithDifferentTypes(thrust::hpx::unseq.on(exec));
}
DECLARE_UNITTEST(TestFindUnseqOnSequencedExecutor);

void TestFindUnseqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestFind(thrust::hpx::unseq.on(exec));
  TestFindWithDifferentTypes(thrust::hpx::unseq.on(exec));
}
DECLARE_UNITTEST(TestFindUnseqOnTestSyncExecutor);

void TestFindUnseqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestFind(thrust::hpx::unseq.with(acs));
  TestFindWithDifferentTypes(thrust::hpx::unseq.with(acs));
}
DECLARE_UNITTEST(TestFindUnseqWithAutoChunkSize);

_CCCL_DIAG_POP
