#include <algorithm>
#include <cstddef>
#include <execution>
#include <numeric>
#include <vector>

int main()
{
  constexpr std::size_t N = 1 << 16;

  auto all_one = [](const int val) {
    return val == 1;
  };

  std::vector<int> in(N);
  std::vector<int> out(N);
  std::iota(in.begin(), in.end(), 0);

  // Default op (difference)
  std::adjacent_difference(std::execution::par, in.begin(), in.end(), out.begin());

  if (out[0] != in[0])
  {
    return 1;
  }
  if (!std::all_of(out.begin() + 1, out.end(), all_one))
  {
    return 1;
  }

  // Custom binary op: sum of neighbors: out[0] = in[0]
  std::fill(out.begin(), out.end(), 0);
  std::adjacent_difference(std::execution::par, in.begin(), in.end(), out.begin(), [](int x, int y) {
    return x + y;
  });

  if (out[0] != in[0])
  {
    return 1;
  }

  for (std::size_t i = 1; i < N; ++i)
  {
    const int expected = in[i] + in[i - 1];
    if (out[i] != expected)
    {
      return 1;
    }
  }

  return 0;
}
