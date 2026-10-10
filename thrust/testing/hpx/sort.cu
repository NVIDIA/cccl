#include <thrust/sort.h>

#include "test_executor.h"
#include <unittest/unittest.h>

template <typename ExecutionPolicy>
void TestSort(ExecutionPolicy exec)
{
  using Vector = thrust::device_vector<int>;
  using T      = typename Vector::value_type;

  {
    Vector data{5, 2, 8, 1, 9, 3};
    Vector ref{1, 2, 3, 5, 8, 9};

    thrust::sort(exec, data.begin(), data.end());

    REQUIRE(data == ref);
  }
  {
    Vector data{5, 2, 8, 1, 9, 3};
    Vector ref{9, 8, 5, 3, 2, 1};

    thrust::sort(exec, data.begin(), data.end(), thrust::greater<T>());

    REQUIRE(data == ref);
  }
}

TEST_CASE("TestSortPar", "[hpx]")
{
  TestSort(thrust::hpx::par);
}

TEST_CASE("TestSortParOnSequencedExecutor", "[hpx]")
{
  hpx::execution::sequenced_executor exec{};

  TestSort(thrust::hpx::par.on(exec));
}

TEST_CASE("TestSortParOnTestSyncExecutor", "[hpx]")
{
  test_sync_executor exec{};

  TestSort(thrust::hpx::par.on(exec));
}

TEST_CASE("TestSortParWithAutoChunkSize", "[hpx]")
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestSort(thrust::hpx::par.with(acs));
}

TEST_CASE("TestSortParUnseq", "[hpx]")
{
  TestSort(thrust::hpx::par_unseq);
}

TEST_CASE("TestSortParUnseqOnSequencedExecutor", "[hpx]")
{
  hpx::execution::sequenced_executor exec{};

  TestSort(thrust::hpx::par_unseq.on(exec));
}

TEST_CASE("TestSortParUnseqOnTestSyncExecutor", "[hpx]")
{
  test_sync_executor exec{};

  TestSort(thrust::hpx::par_unseq.on(exec));
}

TEST_CASE("TestSortParUnseqWithAutoChunkSize", "[hpx]")
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestSort(thrust::hpx::par_unseq.with(acs));
}

TEST_CASE("TestSortSeq", "[hpx]")
{
  TestSort(thrust::hpx::seq);
}

TEST_CASE("TestSortSeqOnSequencedExecutor", "[hpx]")
{
  hpx::execution::sequenced_executor exec{};

  TestSort(thrust::hpx::seq.on(exec));
}

TEST_CASE("TestSortSeqOnTestSyncExecutor", "[hpx]")
{
  test_sync_executor exec{};

  TestSort(thrust::hpx::seq.on(exec));
}

TEST_CASE("TestSortSeqWithAutoChunkSize", "[hpx]")
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestSort(thrust::hpx::seq.with(acs));
}

TEST_CASE("TestSortUnseq", "[hpx]")
{
  TestSort(thrust::hpx::unseq);
}

TEST_CASE("TestSortUnseqOnSequencedExecutor", "[hpx]")
{
  hpx::execution::sequenced_executor exec{};

  TestSort(thrust::hpx::unseq.on(exec));
}

TEST_CASE("TestSortUnseqOnTestSyncExecutor", "[hpx]")
{
  test_sync_executor exec{};

  TestSort(thrust::hpx::unseq.on(exec));
}

TEST_CASE("TestSortUnseqWithAutoChunkSize", "[hpx]")
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestSort(thrust::hpx::unseq.with(acs));
}
