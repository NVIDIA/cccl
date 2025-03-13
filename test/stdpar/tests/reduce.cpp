#include <algorithm>
#include <cassert>
#include <execution>
#include <numeric>
#include <vector>

constexpr int N = 10000;

int main()
{
  std::vector<int> v(N);
  std::fill(std::execution::par_unseq, v.begin(), v.end(), 42);
  int sum = std::reduce(std::execution::par_unseq, v.begin(), v.end(), 100, [](int a, int b) {
    return a + b;
  });
  assert(sum == (42 * N) + 100);
}
