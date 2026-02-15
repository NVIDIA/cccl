#include <thrust/for_each.h>

#include "test_executor.h"
#include <unittest/unittest.h>

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4244 4267) // possible loss of data

template <typename T>
class mark_present_for_each
{
public:
  T* ptr;
  _CCCL_HOST_DEVICE void operator()(T x)
  {
    ptr[(int) x] = 1;
  }
};

template <typename ExecutionPolicy>
void TestForEach(ExecutionPolicy exec)
{
  using Vector = thrust::device_vector<int>;
  using T      = typename Vector::value_type;

  Vector input{3, 2, 3, 4, 6};
  Vector ref{0, 0, 1, 1, 1, 0, 1};

  mark_present_for_each<T> f;

  {
    Vector output(7, (T) 0);
    f.ptr = thrust::raw_pointer_cast(output.data());

    typename Vector::iterator result = thrust::for_each(exec, input.begin(), input.end(), f);

    ASSERT_EQUAL(output, ref);
    ASSERT_EQUAL_QUIET(result, input.end());
  }

  {
    Vector output(7, (T) 0);
    f.ptr = thrust::raw_pointer_cast(output.data());

    typename Vector::iterator result = thrust::for_each_n(exec, input.begin(), input.size(), f);

    ASSERT_EQUAL(output, ref);
    ASSERT_EQUAL_QUIET(result, input.end());
  }
}

void TestForEachPar()
{
  TestForEach(thrust::hpx::par);
}
DECLARE_UNITTEST(TestForEachPar);

void TestForEachParOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestForEach(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestForEachParOnSequencedExecutor);

void TestForEachParOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestForEach(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestForEachParOnTestSyncExecutor);

void TestForEachParWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestForEach(thrust::hpx::par.with(acs));
}
DECLARE_UNITTEST(TestForEachParWithAutoChunkSize);

void TestForEachParUnseq()
{
  TestForEach(thrust::hpx::par_unseq);
}
DECLARE_UNITTEST(TestForEachParUnseq);

void TestForEachParUnseqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestForEach(thrust::hpx::par_unseq.on(exec));
}
DECLARE_UNITTEST(TestForEachParUnseqOnSequencedExecutor);

void TestForEachParUnseqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestForEach(thrust::hpx::par_unseq.on(exec));
}
DECLARE_UNITTEST(TestForEachParUnseqOnTestSyncExecutor);

void TestForEachParUnseqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestForEach(thrust::hpx::par_unseq.with(acs));
}
DECLARE_UNITTEST(TestForEachParUnseqWithAutoChunkSize);

void TestForEachSeq()
{
  TestForEach(thrust::hpx::seq);
}
DECLARE_UNITTEST(TestForEachSeq);

void TestForEachSeqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestForEach(thrust::hpx::seq.on(exec));
}
DECLARE_UNITTEST(TestForEachSeqOnSequencedExecutor);

void TestForEachSeqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestForEach(thrust::hpx::seq.on(exec));
}
DECLARE_UNITTEST(TestForEachSeqOnTestSyncExecutor);

void TestForEachSeqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestForEach(thrust::hpx::seq.with(acs));
}
DECLARE_UNITTEST(TestForEachSeqWithAutoChunkSize);

void TestForEachUnseq()
{
  TestForEach(thrust::hpx::unseq);
}
DECLARE_UNITTEST(TestForEachUnseq);

void TestForEachUnseqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestForEach(thrust::hpx::unseq.on(exec));
}
DECLARE_UNITTEST(TestForEachUnseqOnSequencedExecutor);

void TestForEachUnseqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestForEach(thrust::hpx::unseq.on(exec));
}
DECLARE_UNITTEST(TestForEachUnseqOnTestSyncExecutor);

void TestForEachUnseqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestForEach(thrust::hpx::unseq.with(acs));
}
DECLARE_UNITTEST(TestForEachUnseqWithAutoChunkSize);

_CCCL_DIAG_POP
