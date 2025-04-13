#include <algorithm>
#include <cassert>
#include <execution>
#include <numeric>
#include <vector>

// Ensure that we are indeed using the correct CCCL version
static_assert(CCCL_MAJOR_VERSION == CMAKE_CCCL_VERSION_MAJOR);
static_assert(CCCL_MINOR_VERSION == CMAKE_CCCL_VERSION_MINOR);
static_assert(CCCL_PATCH_VERSION == CMAKE_CCCL_VERSION_PATCH);

constexpr int N = 1000;

int main()
{
  std::vector<int> v(N);
  std::fill(std::execution::par_unseq, v.begin(), v.end(), 42);

  int sum = std::reduce(std::execution::par_unseq, v.begin(), v.end(), 100, [](int a, int b) {
    return a + b;
  });
  assert(sum == (42 * N) + 100);

  sum = std::reduce(std::execution::par_unseq, v.begin(), v.end(), 100);
  assert(sum == (42 * N) + 100);
}
