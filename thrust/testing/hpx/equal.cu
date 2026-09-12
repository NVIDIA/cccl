#include <thrust/equal.h>

#include "test_executor.h"
#include <unittest/unittest.h>

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4244 4267) // possible loss of data

template <typename ExecutionPolicy>
void TestEqual(ExecutionPolicy exec)
{
  using Vector = thrust::device_vector<int>;
  using T      = typename Vector::value_type;

  Vector v1{5, 2, 0, 0, 0};
  Vector v2{5, 2, 0, 6, 1};

  {
    ASSERT_EQUAL(thrust::equal(v1.begin(), v1.end(), v1.begin()), true);
    ASSERT_EQUAL(thrust::equal(v1.begin(), v1.end(), v2.begin()), false);

    ASSERT_EQUAL(thrust::equal(v1.begin(), v1.begin() + 0, v1.begin()), true);
    ASSERT_EQUAL(thrust::equal(v1.begin(), v1.begin() + 1, v1.begin()), true);
    ASSERT_EQUAL(thrust::equal(v1.begin(), v1.begin() + 3, v2.begin()), true);
    ASSERT_EQUAL(thrust::equal(v1.begin(), v1.begin() + 4, v2.begin()), false);
  }
  {
    ASSERT_EQUAL(thrust::equal(v1.begin(), v1.end(), v2.begin(), thrust::less_equal<T>()), true);
    ASSERT_EQUAL(thrust::equal(v1.begin(), v1.end(), v2.begin(), thrust::greater<T>()), false);
  }
}

void TestEqualPar()
{
  TestEqual(thrust::hpx::par);
}
DECLARE_UNITTEST(TestEqualPar);

void TestEqualParOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestEqual(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestEqualParOnSequencedExecutor);

void TestEqualParOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestEqual(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestEqualParOnTestSyncExecutor);

void TestEqualParWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestEqual(thrust::hpx::par.with(acs));
}
DECLARE_UNITTEST(TestEqualParWithAutoChunkSize);

void TestEqualParUnseq()
{
  TestEqual(thrust::hpx::par_unseq);
}
DECLARE_UNITTEST(TestEqualParUnseq);

void TestEqualParUnseqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestEqual(thrust::hpx::par_unseq.on(exec));
}
DECLARE_UNITTEST(TestEqualParUnseqOnSequencedExecutor);

void TestEqualParUnseqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestEqual(thrust::hpx::par_unseq.on(exec));
}
DECLARE_UNITTEST(TestEqualParUnseqOnTestSyncExecutor);

void TestEqualParUnseqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestEqual(thrust::hpx::par_unseq.with(acs));
}
DECLARE_UNITTEST(TestEqualParUnseqWithAutoChunkSize);

void TestEqualSeq()
{
  TestEqual(thrust::hpx::seq);
}
DECLARE_UNITTEST(TestEqualSeq);

void TestEqualSeqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestEqual(thrust::hpx::seq.on(exec));
}
DECLARE_UNITTEST(TestEqualSeqOnSequencedExecutor);

void TestEqualSeqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestEqual(thrust::hpx::seq.on(exec));
}
DECLARE_UNITTEST(TestEqualSeqOnTestSyncExecutor);

void TestEqualSeqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestEqual(thrust::hpx::seq.with(acs));
}
DECLARE_UNITTEST(TestEqualSeqWithAutoChunkSize);

void TestEqualUnseq()
{
  TestEqual(thrust::hpx::unseq);
}
DECLARE_UNITTEST(TestEqualUnseq);

void TestEqualUnseqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestEqual(thrust::hpx::unseq.on(exec));
}
DECLARE_UNITTEST(TestEqualUnseqOnSequencedExecutor);

void TestEqualUnseqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestEqual(thrust::hpx::unseq.on(exec));
}
DECLARE_UNITTEST(TestEqualUnseqOnTestSyncExecutor);

void TestEqualUnseqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestEqual(thrust::hpx::unseq.with(acs));
}
DECLARE_UNITTEST(TestEqualUnseqWithAutoChunkSize);

_CCCL_DIAG_POP
