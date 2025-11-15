#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/system/cuda/vector.h>

#include <cuda/std/utility>

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>

// This example demonstrates how to control how Thrust allocates temporary
// storage during algorithms such as thrust::sort. The idea will be to create a
// simple cache of allocations to search when temporary storage is requested.
// If a hit is found in the cache, we quickly return the cached allocation
// instead of resorting to the more expensive thrust::cuda::malloc.

// Note: Thrust now has its own caching allocator layer; if you just need a
// caching allocator, you ought to use that. This example is still useful
// as a demonstration of how to use a Thrust custom allocator.

// Note: this implementation cached_allocator is not thread-safe. If multiple
// (host) threads use the same cached_allocator then they should gain exclusive
// access to the allocator before accessing its methods.

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

// A simple allocator for caching cudaMalloc allocations.
// A minimum allocator needs to provide at least `value_type`, `allocate` and `deallocate`.
struct cached_allocator
{
  using value_type = char;

  ~cached_allocator()
  {
    free_all();
  }

  char* allocate(std::ptrdiff_t num_bytes)
  {
    std::cout << "cached_allocator::allocate(): num_bytes == " << num_bytes << std::endl;

    char* result = nullptr;

    // Search the cache for a free block.
    auto free_block_it = free_blocks.find(num_bytes);
    if (free_block_it != free_blocks.end())
    {
      std::cout << "cached_allocator::allocate(): found a free block" << std::endl;
      result = free_block_it->second;
      free_blocks.erase(free_block_it);
    }
    else
    {
      // No allocation of the right size exists, so create a new one with `thrust::cuda::malloc`.
      std::cout << "cached_allocator::allocate(): allocating new block" << std::endl;
      // Allocate memory and convert the resulting `thrust::cuda::pointer` to a raw pointer.
      result = thrust::cuda::malloc<char>(num_bytes).get();
    }

    // Insert the allocated pointer into the `allocated_blocks` map.
    allocated_blocks.insert(std::pair{result, num_bytes});

    return result;
  }

  void deallocate(char* ptr, size_t)
  {
    std::cout << "cached_allocator::deallocate(): ptr == " << reinterpret_cast<void*>(ptr) << std::endl;

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
    std::cout << "cached_allocator::free_all()" << std::endl;

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

int main()
{
  std::size_t num_elements = 32768;

  thrust::host_vector<int> h_input(num_elements);

  // Generate random input.
  thrust::generate(h_input.begin(), h_input.end(), rand);

  thrust::cuda::vector<int> d_input = h_input;
  thrust::cuda::vector<int> d_result(num_elements);

  std::size_t num_trials = 5;

  cached_allocator alloc;

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
