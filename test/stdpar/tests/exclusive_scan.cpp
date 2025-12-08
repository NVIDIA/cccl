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
    in[i] = static_cast<int>(i + 1);
  }

  // exclusive_scan with default initial value (0)
  std::exclusive_scan(std::execution::par, in.begin(), in.end(), out.begin(), 0);

  int running_sum = 0;
  for (std::size_t i = 0; i < N; ++i)
  {
    if (out[i] != running_sum)
    {
      return 1;
    }
    running_sum += in[i];
  }

  // exclusive_scan with non-zero initial value
  const int init = 42;
  std::exclusive_scan(std::execution::par, in.begin(), in.end(), out.begin(), init);

  running_sum = init;
  for (std::size_t i = 0; i < N; ++i)
  {
    if (out[i] != running_sum)
    {
      return 1;
    }
    running_sum += in[i];
  }

  return 0;
}
