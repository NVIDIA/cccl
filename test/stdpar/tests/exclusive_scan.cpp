#include <cstddef>
#include <execution>
#include <functional>
#include <numeric>
#include <vector>

int main()
{
  constexpr std::size_t N = 1 << 16;

  std::vector<int> in(N);
  std::vector<int> out(N);

  std::iota(in.begin(), in.end(), 1);

  // exclusive_scan with non-zero initial value
  const int init = 42;
  std::exclusive_scan(std::execution::par, in.begin(), in.end(), out.begin(), init);

  int running_sum = init;
  for (std::size_t i = 0; i < N; ++i)
  {
    if (out[i] != running_sum)
    {
      return 1;
    }
    running_sum += in[i];
  }

  // exclusive_scan with initial value & binary ops
  const int mul_init = 1;
  std::exclusive_scan(std::execution::par, in.begin(), in.end(), out.begin(), mul_init, std::multiplies<int>{});

  int expected = mul_init;
  for (std::size_t i = 0; i < N; ++i)
  {
    if (out[i] != expected)
    {
      return 1;
    }
    expected *= 2;
  }

  // exclusive_scan with initial value & custom binary ops
  const int init2 = 7;
  std::exclusive_scan(
    std::execution::par,
    in.begin(),
    in.end(),
    out.begin(),
    init2,
    [](int lhs, int rhs) {
      return lhs + rhs + 1;
    })

    for (std::size_t i = 0; i < N; ++i)
  {
    const int expected = init2 + static_cast<int>(2 * i);
    if (out[i] != expected)
    {
      return 1;
    }
  }

  return 0;
}
