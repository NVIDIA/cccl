#include <cstddef>
#include <execution>
#include <numeric>
#include <vector>

int main()
{
  constexpr std::size_t N = 1 << 16;

  std::vector<int> in(N);
  std::vector<int> out(N);

  for (std::size_t i = 0; i < N; ++i)
  {
    in[i] = 1;
  }

  // inclusive_scan with default initial value
  std::inclusive_scan(std::execution::par, in.begin(), in.end(), out.begin());

  for (std::size_t i = 0; i < N; ++i)
  {
    const int expected = static_cast<int>(i + 1);
    if (out[i] != expected)
    {
      return 1;
    }
  }

  // inclusive_scan with non-zero initial value
  const int init = 42;
  std::inclusive_scan(std::execution::par, in.begin(), in.end(), out.begin(), init);

  for (std::size_t i = 0; i < N; ++i)
  {
    const int expected = init + static_cast<int>(i + 1);
    if (out[i] != expected)
    {
      return 1;
    }
  }

  return 0;
}
