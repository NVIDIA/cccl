#include <thrust/fill.h>

#include "test_executor.h"
#include <unittest/unittest.h>

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4244 4267) // possible loss of data

template <typename ExecutionPolicy>
void TestFill(ExecutionPolicy exec)
{
  using Vector = thrust::device_vector<int>;
  using T      = typename Vector::value_type;

  Vector ref{1, 1, 1, 1, 1};

  {
    Vector v{0, 1, 2, 3, 4};

    thrust::fill(v.begin(), v.end(), (T) 1);

    ASSERT_EQUAL(v, ref);
  }
  {
    Vector v{0, 1, 2, 3, 4};

    thrust::fill_n(v.begin(), v.size(), (T) 1);

    ASSERT_EQUAL(v, ref);
  }
}

void TestFillPar()
{
  TestFill(thrust::hpx::par);
}
DECLARE_UNITTEST(TestFillPar);

void TestFillParOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestFill(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestFillParOnSequencedExecutor);

void TestFillParOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestFill(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestFillParOnTestSyncExecutor);

void TestFillParWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestFill(thrust::hpx::par.with(acs));
}
DECLARE_UNITTEST(TestFillParWithAutoChunkSize);

void TestFillParUnseq()
{
  TestFill(thrust::hpx::par_unseq);
}
DECLARE_UNITTEST(TestFillParUnseq);

void TestFillParUnseqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestFill(thrust::hpx::par_unseq.on(exec));
}
DECLARE_UNITTEST(TestFillParUnseqOnSequencedExecutor);

void TestFillParUnseqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestFill(thrust::hpx::par_unseq.on(exec));
}
DECLARE_UNITTEST(TestFillParUnseqOnTestSyncExecutor);

void TestFillParUnseqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestFill(thrust::hpx::par_unseq.with(acs));
}
DECLARE_UNITTEST(TestFillParUnseqWithAutoChunkSize);

void TestFillSeq()
{
  TestFill(thrust::hpx::seq);
}
DECLARE_UNITTEST(TestFillSeq);

void TestFillSeqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestFill(thrust::hpx::seq.on(exec));
}
DECLARE_UNITTEST(TestFillSeqOnSequencedExecutor);

void TestFillSeqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestFill(thrust::hpx::seq.on(exec));
}
DECLARE_UNITTEST(TestFillSeqOnTestSyncExecutor);

void TestFillSeqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestFill(thrust::hpx::seq.with(acs));
}
DECLARE_UNITTEST(TestFillSeqWithAutoChunkSize);

void TestFillUnseq()
{
  TestFill(thrust::hpx::unseq);
}
DECLARE_UNITTEST(TestFillUnseq);

void TestFillUnseqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestFill(thrust::hpx::unseq.on(exec));
}
DECLARE_UNITTEST(TestFillUnseqOnSequencedExecutor);

void TestFillUnseqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestFill(thrust::hpx::unseq.on(exec));
}
DECLARE_UNITTEST(TestFillUnseqOnTestSyncExecutor);

void TestFillUnseqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestFill(thrust::hpx::unseq.with(acs));
}
DECLARE_UNITTEST(TestFillUnseqWithAutoChunkSize);

_CCCL_DIAG_POP
