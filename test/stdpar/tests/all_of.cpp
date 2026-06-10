#include <algorithm>
#include <cstddef>
#include <execution>
#include <vector>

// Ensure that we are indeed using the correct CCCL version
static_assert(CCCL_MAJOR_VERSION == CMAKE_CCCL_VERSION_MAJOR);
static_assert(CCCL_MINOR_VERSION == CMAKE_CCCL_VERSION_MINOR);
static_assert(CCCL_PATCH_VERSION == CMAKE_CCCL_VERSION_PATCH);

int main()
{
  constexpr std::size_t N = 1 << 16;
  std::vector<int> v(N, 1);

  // All elements are 1, so this should be true
  const bool all_ones = std::all_of(std::execution::par, v.begin(), v.end(), [](int x) {
    return x == 1;
  });

  if (!all_ones)
  {
    return 1;
  }

  // Flip a single element in the middle, now all elements are not 1
  v[N / 2]                  = 2;
  const bool still_all_ones = std::all_of(std::execution::par, v.begin(), v.end(), [](int x) {
    return x == 1;
  });

  if (still_all_ones)
  {
    return 1;
  }

  return 0;
}
