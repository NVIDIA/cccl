#include <thrust/copy.h>

#include "test_executor.h"
#include <unittest/unittest.h>

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4244 4267) // possible loss of data

template <typename ExecutionPolicy>
void TestCopy(ExecutionPolicy exec)
{
  using Vector = thrust::device_vector<int>;
  using T      = typename Vector::value_type;

  Vector ref{0, 1, 2, 3, 4};

  {
    Vector v{1, 1, 1, 1, 1};

    thrust::copy(ref.cbegin(), ref.cend(), v.begin());

    ASSERT_EQUAL(v, ref);
  }
  {
    Vector v{1, 1, 1, 1, 1};

    thrust::copy_n(ref.cbegin(), ref.size(), v.begin());

    ASSERT_EQUAL(v, ref);
  }
}

void TestCopyPar()
{
  TestCopy(thrust::hpx::par);
}
DECLARE_UNITTEST(TestCopyPar);

void TestCopyParOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestCopy(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestCopyParOnSequencedExecutor);

void TestCopyParOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestCopy(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestCopyParOnTestSyncExecutor);

void TestCopyParWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestCopy(thrust::hpx::par.with(acs));
}
DECLARE_UNITTEST(TestCopyParWithAutoChunkSize);

void TestCopyParUnseq()
{
  TestCopy(thrust::hpx::par_unseq);
}
DECLARE_UNITTEST(TestCopyParUnseq);

void TestCopyParUnseqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestCopy(thrust::hpx::par_unseq.on(exec));
}
DECLARE_UNITTEST(TestCopyParUnseqOnSequencedExecutor);

void TestCopyParUnseqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestCopy(thrust::hpx::par_unseq.on(exec));
}
DECLARE_UNITTEST(TestCopyParUnseqOnTestSyncExecutor);

void TestCopyParUnseqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestCopy(thrust::hpx::par_unseq.with(acs));
}
DECLARE_UNITTEST(TestCopyParUnseqWithAutoChunkSize);

void TestCopySeq()
{
  TestCopy(thrust::hpx::seq);
}
DECLARE_UNITTEST(TestCopySeq);

void TestCopySeqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestCopy(thrust::hpx::seq.on(exec));
}
DECLARE_UNITTEST(TestCopySeqOnSequencedExecutor);

void TestCopySeqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestCopy(thrust::hpx::seq.on(exec));
}
DECLARE_UNITTEST(TestCopySeqOnTestSyncExecutor);

void TestCopySeqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestCopy(thrust::hpx::seq.with(acs));
}
DECLARE_UNITTEST(TestCopySeqWithAutoChunkSize);

void TestCopyUnseq()
{
  TestCopy(thrust::hpx::unseq);
}
DECLARE_UNITTEST(TestCopyUnseq);

void TestCopyUnseqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestCopy(thrust::hpx::unseq.on(exec));
}
DECLARE_UNITTEST(TestCopyUnseqOnSequencedExecutor);

void TestCopyUnseqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestCopy(thrust::hpx::unseq.on(exec));
}
DECLARE_UNITTEST(TestCopyUnseqOnTestSyncExecutor);

void TestCopyUnseqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestCopy(thrust::hpx::unseq.with(acs));
}
DECLARE_UNITTEST(TestCopyUnseqWithAutoChunkSize);

_CCCL_DIAG_POP
