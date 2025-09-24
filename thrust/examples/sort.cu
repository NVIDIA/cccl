#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <cuda/std/utility>

#include <iomanip>
#include <iostream>
#include <numeric>

// Helper routines

void initialize(thrust::device_vector<int>& v)
{
  thrust::default_random_engine rng(123456);
  thrust::uniform_int_distribution<int> dist(10, 99);
  thrust::host_vector<int> host_data(v.size());
  for (auto& e : host_data)
  {
    e = dist(rng);
  }
  v = host_data;
}

void initialize(thrust::device_vector<float>& v)
{
  thrust::default_random_engine rng(123456);
  thrust::uniform_int_distribution<int> dist(2, 19);
  thrust::host_vector<float> host_data(v.size());
  for (auto& e : host_data)
  {
    e = dist(rng) / 2.0f;
  }
  v = host_data;
}

void initialize(thrust::device_vector<cuda::std::pair<int, int>>& v)
{
  thrust::default_random_engine rng(123456);
  thrust::uniform_int_distribution<int> dist(0, 9);
  thrust::host_vector<cuda::std::pair<int, int>> host_data(v.size());
  for (auto& e : host_data)
  {
    int a = dist(rng);
    int b = dist(rng);
    e     = cuda::std::make_pair(a, b);
  }
  v = host_data;
}

void initialize(thrust::device_vector<int>& v1, thrust::device_vector<int>& v2)
{
  thrust::default_random_engine rng(123456);
  thrust::uniform_int_distribution<int> dist(10, 99);
  thrust::host_vector<int> host_data(v1.size());
  for (auto& e : host_data)
  {
    e = dist(rng);
  }
  v1 = host_data;
  thrust::sequence(v2.begin(), v2.end(), 0);
}

void print(const thrust::device_vector<int>& v)
{
  for (const auto& value : v)
  {
    std::cout << " " << value;
  }
  std::cout << "\n";
}

void print(const thrust::device_vector<float>& v)
{
  for (const auto& value : v)
  {
    std::cout << " " << std::fixed << std::setprecision(1) << value;
  }
  std::cout << "\n";
}

void print(const thrust::device_vector<cuda::std::pair<int, int>>& v)
{
  for (const auto& p : v)
  {
    cuda::std::pair<int, int> local_p = p;
    std::cout << " (" << local_p.first << "," << local_p.second << ")";
  }
  std::cout << "\n";
}

void print(thrust::device_vector<int>& v1, thrust::device_vector<int> v2)
{
  for (size_t i = 0; i < v1.size(); i++)
  {
    std::cout << " (" << v1[i] << "," << std::setw(2) << v2[i] << ")";
  }
  std::cout << "\n";
}

// user-defined comparison operator that acts like less<int>,
// except even numbers are considered to be smaller than odd numbers
struct evens_before_odds
{
  __host__ __device__ bool operator()(int x, int y)
  {
    if (x % 2 == y % 2)
    {
      return x < y;
    }
    else if (x % 2)
    {
      return false;
    }
    else
    {
      return true;
    }
  }
};

int main()
{
  size_t N = 16;

  std::cout << "sorting integers\n";
  {
    thrust::device_vector<int> keys(N);
    initialize(keys);
    print(keys);
    thrust::sort(keys.begin(), keys.end());
    print(keys);
  }

  std::cout << "\nsorting integers (descending)\n";
  {
    thrust::device_vector<int> keys(N);
    initialize(keys);
    print(keys);
    thrust::sort(keys.begin(), keys.end(), ::cuda::std::greater<int>());
    print(keys);
  }

  std::cout << "\nsorting integers (user-defined comparison)\n";
  {
    thrust::device_vector<int> keys(N);
    initialize(keys);
    print(keys);
    thrust::sort(keys.begin(), keys.end(), evens_before_odds());
    print(keys);
  }

  std::cout << "\nsorting floats\n";
  {
    thrust::device_vector<float> keys(N);
    initialize(keys);
    print(keys);
    thrust::sort(keys.begin(), keys.end());
    print(keys);
  }

  std::cout << "\nsorting pairs\n";
  {
    thrust::device_vector<cuda::std::pair<int, int>> keys(N);
    initialize(keys);
    print(keys);
    thrust::sort(keys.begin(), keys.end());
    print(keys);
  }

  std::cout << "\nkey-value sorting\n";
  {
    thrust::device_vector<int> keys(N);
    thrust::device_vector<int> values(N);
    initialize(keys, values);
    print(keys, values);
    thrust::sort_by_key(keys.begin(), keys.end(), values.begin());
    print(keys, values);
  }

  std::cout << "\nkey-value sorting (descending)\n";
  {
    thrust::device_vector<int> keys(N);
    thrust::device_vector<int> values(N);
    initialize(keys, values);
    print(keys, values);
    thrust::sort_by_key(keys.begin(), keys.end(), values.begin(), ::cuda::std::greater<int>());
    print(keys, values);
  }

  return 0;
}
