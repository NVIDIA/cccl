#include <thrust/count.h>

#include "test_executor.h"
#include <unittest/unittest.h>

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4244 4267) // possible loss of data

template <typename ExecutionPolicy>
void TestCount(ExecutionPolicy exec)
{
  using Vector = thrust::device_vector<int>;
  using T      = typename Vector::value_type;

  Vector data{1, 6, 1, 9, 2};

  {
    ASSERT_EQUAL(thrust::count(data.begin(), data.end(), 0), 0);
    ASSERT_EQUAL(thrust::count(data.begin(), data.end(), 1), 2);
    ASSERT_EQUAL(thrust::count(data.begin(), data.end(), 2), 1);
  }
  {
    auto greater_than_five = [](const T& x) {
      return x > 5;
    };
    ASSERT_EQUAL(thrust::count_if(data.begin(), data.end(), greater_than_five), 2);
  }
}

void TestCountPar()
{
  TestCount(thrust::hpx::par);
}
DECLARE_UNITTEST(TestCountPar);

void TestCountParOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestCount(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestCountParOnSequencedExecutor);

void TestCountParOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestCount(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestCountParOnTestSyncExecutor);

void TestCountParWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestCount(thrust::hpx::par.with(acs));
}
DECLARE_UNITTEST(TestCountParWithAutoChunkSize);

void TestCountParUnseq()
{
  TestCount(thrust::hpx::par_unseq);
}
DECLARE_UNITTEST(TestCountParUnseq);

void TestCountParUnseqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestCount(thrust::hpx::par_unseq.on(exec));
}
DECLARE_UNITTEST(TestCountParUnseqOnSequencedExecutor);

void TestCountParUnseqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestCount(thrust::hpx::par_unseq.on(exec));
}
DECLARE_UNITTEST(TestCountParUnseqOnTestSyncExecutor);

void TestCountParUnseqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestCount(thrust::hpx::par_unseq.with(acs));
}
DECLARE_UNITTEST(TestCountParUnseqWithAutoChunkSize);

void TestCountSeq()
{
  TestCount(thrust::hpx::seq);
}
DECLARE_UNITTEST(TestCountSeq);

void TestCountSeqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestCount(thrust::hpx::seq.on(exec));
}
DECLARE_UNITTEST(TestCountSeqOnSequencedExecutor);

void TestCountSeqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestCount(thrust::hpx::seq.on(exec));
}
DECLARE_UNITTEST(TestCountSeqOnTestSyncExecutor);

void TestCountSeqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestCount(thrust::hpx::seq.with(acs));
}
DECLARE_UNITTEST(TestCountSeqWithAutoChunkSize);

void TestCountUnseq()
{
  TestCount(thrust::hpx::unseq);
}
DECLARE_UNITTEST(TestCountUnseq);

void TestCountUnseqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestCount(thrust::hpx::unseq.on(exec));
}
DECLARE_UNITTEST(TestCountUnseqOnSequencedExecutor);

void TestCountUnseqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestCount(thrust::hpx::unseq.on(exec));
}
DECLARE_UNITTEST(TestCountUnseqOnTestSyncExecutor);

void TestCountUnseqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestCount(thrust::hpx::unseq.with(acs));
}
DECLARE_UNITTEST(TestCountUnseqWithAutoChunkSize);

_CCCL_DIAG_POP
