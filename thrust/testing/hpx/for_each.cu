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

    REQUIRE(output == ref);
    REQUIRE(result == input.end());
  }

  {
    Vector output(7, (T) 0);
    f.ptr = thrust::raw_pointer_cast(output.data());

    typename Vector::iterator result = thrust::for_each_n(exec, input.begin(), input.size(), f);

    REQUIRE(output == ref);
    REQUIRE(result == input.end());
  }
}

TEST_CASE("TestForEachPar", "[hpx]")
{
  TestForEach(thrust::hpx::par);
}

TEST_CASE("TestForEachParOnSequencedExecutor", "[hpx]")
{
  hpx::execution::sequenced_executor exec{};

  TestForEach(thrust::hpx::par.on(exec));
}

TEST_CASE("TestForEachParOnTestSyncExecutor", "[hpx]")
{
  test_sync_executor exec{};

  TestForEach(thrust::hpx::par.on(exec));
}

TEST_CASE("TestForEachParWithAutoChunkSize", "[hpx]")
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestForEach(thrust::hpx::par.with(acs));
}

TEST_CASE("TestForEachParUnseq", "[hpx]")
{
  TestForEach(thrust::hpx::par_unseq);
}

TEST_CASE("TestForEachParUnseqOnSequencedExecutor", "[hpx]")
{
  hpx::execution::sequenced_executor exec{};

  TestForEach(thrust::hpx::par_unseq.on(exec));
}

TEST_CASE("TestForEachParUnseqOnTestSyncExecutor", "[hpx]")
{
  test_sync_executor exec{};

  TestForEach(thrust::hpx::par_unseq.on(exec));
}

TEST_CASE("TestForEachParUnseqWithAutoChunkSize", "[hpx]")
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestForEach(thrust::hpx::par_unseq.with(acs));
}

TEST_CASE("TestForEachSeq", "[hpx]")
{
  TestForEach(thrust::hpx::seq);
}

TEST_CASE("TestForEachSeqOnSequencedExecutor", "[hpx]")
{
  hpx::execution::sequenced_executor exec{};

  TestForEach(thrust::hpx::seq.on(exec));
}

TEST_CASE("TestForEachSeqOnTestSyncExecutor", "[hpx]")
{
  test_sync_executor exec{};

  TestForEach(thrust::hpx::seq.on(exec));
}

TEST_CASE("TestForEachSeqWithAutoChunkSize", "[hpx]")
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestForEach(thrust::hpx::seq.with(acs));
}

TEST_CASE("TestForEachUnseq", "[hpx]")
{
  TestForEach(thrust::hpx::unseq);
}

TEST_CASE("TestForEachUnseqOnSequencedExecutor", "[hpx]")
{
  hpx::execution::sequenced_executor exec{};

  TestForEach(thrust::hpx::unseq.on(exec));
}

TEST_CASE("TestForEachUnseqOnTestSyncExecutor", "[hpx]")
{
  test_sync_executor exec{};

  TestForEach(thrust::hpx::unseq.on(exec));
}

TEST_CASE("TestForEachUnseqWithAutoChunkSize", "[hpx]")
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestForEach(thrust::hpx::unseq.with(acs));
}

_CCCL_DIAG_POP
