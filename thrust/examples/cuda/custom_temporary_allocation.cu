#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/system/cuda/memory.h>
#include <thrust/system/cuda/pointer.h>
#include <thrust/system/cuda/vector.h>

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <map>
#include <sstream>
#include <utility>

// This example demonstrates how to control how Thrust allocates temporary
// storage during algorithms such as thrust::sort. The idea will be to create a
// simple cache of allocations to search when temporary storage is requested.
// If a hit is found in the cache, we quickly return the cached allocation
// instead of resorting to the more expensive thrust::cuda::malloc.

// Note: Thrust now has its own caching allocator layer; if you just need a
// caching allocator, you ought to use that. This example is still useful
// as a demonstration of how to use a Thrust custom allocator.

// Note: this implementation of block_cache is not thread-safe. If multiple
// (host) threads use the same block_cache then they should gain exclusive
// access to the cache before accessing its methods.

struct not_my_pointer_exception : std::exception
{
  explicit not_my_pointer_exception(void* p)
  {
    std::stringstream s;
    s << "Pointer `" << p << "` was not allocated by this allocator.";
    message = s.str();
  }

  const char* what() const noexcept override
  {
    return message.c_str();
  }

private:
  std::string message;
};

// A simple cache of cudaMalloc allocations.
struct block_cache
{
  block_cache() = default;

  block_cache(const block_cache&)            = delete;
  block_cache& operator=(const block_cache&) = delete;

  ~block_cache()
  {
    free_all();
  }

  char* allocate(std::ptrdiff_t num_bytes)
  {
    std::cout << "block_cache::allocate(): num_bytes == " << num_bytes << '\n';

    char* result = nullptr;

    // Search the cache for a free block.
    auto free_block_it = free_blocks.find(num_bytes);
    if (free_block_it != free_blocks.end())
    {
      std::cout << "block_cache::allocate(): found a free block" << '\n';
      result = free_block_it->second;
      free_blocks.erase(free_block_it);
    }
    else
    {
      // No allocation of the right size exists, so create a new one with `thrust::cuda::malloc`.
      std::cout << "block_cache::allocate(): allocating new block" << '\n';
      // Allocate memory and convert the resulting `thrust::cuda::pointer` to a raw pointer.
      result = thrust::cuda::malloc<char>(num_bytes).get();
    }

    // Insert the allocated pointer into the `allocated_blocks` map.
    allocated_blocks.insert(std::pair{result, num_bytes});

    return result;
  }

  void deallocate(char* ptr)
  {
    std::cout << "block_cache::deallocate(): ptr == " << reinterpret_cast<void*>(ptr) << '\n';

    // Erase the allocated block from the allocated blocks map.
    auto it = allocated_blocks.find(ptr);
    if (it == allocated_blocks.end())
    {
      throw not_my_pointer_exception(ptr);
    }

    const std::ptrdiff_t num_bytes = it->second;
    allocated_blocks.erase(it);

    // Insert the block into the free blocks map.
    free_blocks.insert(std::make_pair(num_bytes, ptr));
  }

private:
  std::multimap<std::ptrdiff_t, char*> free_blocks;
  std::map<char*, std::ptrdiff_t> allocated_blocks;

  void free_all()
  {
    std::cout << "block_cache::free_all()" << '\n';

    // Deallocate all outstanding blocks in both lists.
    for (auto [bytes, ptr] : free_blocks)
    {
      thrust::cuda::free(thrust::cuda::pointer<char>(ptr));
    }

    for (auto [ptr, bytes] : allocated_blocks)
    {
      thrust::cuda::free(thrust::cuda::pointer<char>(ptr));
    }
  }
};

// A minimum allocator needs to provide at least `value_type`, `allocate` and `deallocate`.
// Thrust rebinds the allocator to the type it allocates, so it is a template and
// can be constructed from an allocator of another type. All copies share one cache.
template <typename T>
struct cached_allocator
{
  using value_type = T;

  explicit cached_allocator(block_cache& c)
      : cache(&c)
  {}

  template <typename U>
  cached_allocator(const cached_allocator<U>& other)
      : cache(other.cache)
  {}

  T* allocate(std::ptrdiff_t n)
  {
    return reinterpret_cast<T*>(cache->allocate(n * static_cast<std::ptrdiff_t>(sizeof(T))));
  }

  void deallocate(T* ptr, size_t)
  {
    cache->deallocate(reinterpret_cast<char*>(ptr));
  }

  block_cache* cache;
};

int main()
{
  const std::size_t num_elements = 32768;

  thrust::host_vector<int> h_input(num_elements);

  // Generate random input.
  thrust::generate(h_input.begin(), h_input.end(), rand);

  const thrust::cuda::vector<int> d_input = h_input;
  thrust::cuda::vector<int> d_result(num_elements);

  const std::size_t num_trials = 5;

  block_cache cache;
  cached_allocator<char> alloc(cache);

  for (std::size_t i = 0; i < num_trials; ++i)
  {
    d_result = d_input;

    // Pass alloc to execution policy cuda::par. It will handle allocations needed inside sort.
    thrust::sort(thrust::cuda::par(alloc), d_result.begin(), d_result.end());

    // Ensure the result is sorted.
    assert(thrust::is_sorted(d_result.begin(), d_result.end()));
  }

  return 0;
}
