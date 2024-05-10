/*

Copyright (c) 2018, NVIDIA Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

*/

#include <cuda/std/atomic>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>

template <class T>
static constexpr T minimum(T a, T b)
{
  return a < b ? a : b;
}

struct trie
{
  struct ref
  {
    cuda::atomic<trie*, cuda::thread_scope_device> ptr = LIBCUDACXX_ATOMIC_VAR_INIT(nullptr);
    // the flag will protect against multiple pointer updates
    cuda::std::atomic_flag flag = LIBCUDACXX_ATOMIC_FLAG_INIT;
  } next[26];
  cuda::std::atomic<short> count = LIBCUDACXX_ATOMIC_VAR_INIT(0);
};
__host__ __device__ int index_of(char c)
{
  if (c >= 'a' && c <= 'z')
  {
    return c - 'a';
  }
  if (c >= 'A' && c <= 'Z')
  {
    return c - 'A';
  }
  return -1;
};
__host__ __device__ void make_trie(
  /* trie to insert word counts into */ trie& root,
  /* bump allocator to get new nodes*/ cuda::std::atomic<trie*>& bump,
  /* input */ const char* begin,
  const char* end,
  /* thread this invocation is for */ unsigned index,
  /* how many threads there are */ unsigned domain)
{
  auto const size   = end - begin;
  auto const stride = (size / domain + 1);

  auto off        = minimum(size, stride * index);
  auto const last = minimum(size, off + stride);

  for (char c = begin[off]; off < size && off != last && c != 0 && index_of(c) != -1; ++off, c = begin[off])
    ;
  for (char c = begin[off]; off < size && off != last && c != 0 && index_of(c) == -1; ++off, c = begin[off])
    ;

  trie* n = &root;
  for (char c = begin[off];; ++off, c = begin[off])
  {
    auto const index = off >= size ? -1 : index_of(c);
    if (index == -1)
    {
      if (n != &root)
      {
        n->count.fetch_add(1, cuda::std::memory_order_relaxed);
        n = &root;
      }
      // end of last word?
      if (off >= size || off > last)
      {
        break;
      }
      else
      {
        continue;
      }
    }
    if (n->next[index].ptr.load(cuda::memory_order_acquire) == nullptr)
    {
      if (n->next[index].flag.test_and_set(cuda::std::memory_order_relaxed))
      {
        n->next[index].ptr.wait(nullptr, cuda::std::memory_order_acquire);
      }
      else
      {
        auto next = bump.fetch_add(1, cuda::std::memory_order_relaxed);
        n->next[index].ptr.store(next, cuda::std::memory_order_release);
        n->next[index].ptr.notify_all();
      }
    }
    n = n->next[index].ptr.load(cuda::std::memory_order_relaxed);
  }
}

__global__ // __launch_bounds__(1024, 1)
  void
  call_make_trie(trie* t, cuda::std::atomic<trie*>* bump, const char* begin, const char* end)
{
  auto const index  = blockDim.x * blockIdx.x + threadIdx.x;
  auto const domain = gridDim.x * blockDim.x;
  make_trie(*t, *bump, begin, end, index, domain);
}

__global__ void do_nothing() {}

#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#define check(ans)                      \
  {                                     \
    assert_((ans), __FILE__, __LINE__); \
  }
inline void assert_(cudaError_t code, const char* file, int line)
{
  if (code == cudaSuccess)
  {
    return;
  }
  std::cerr << "check failed: " << cudaGetErrorString(code) << " : " << file << '@' << line << std::endl;
  abort();
}

template <class T>
struct managed_allocator
{
  typedef cuda::std::size_t size_type;
  typedef cuda::std::ptrdiff_t difference_type;

  typedef T value_type;
  typedef T* pointer; // (deprecated in C++17)(removed in C++20) T*
  typedef const T* const_pointer; // (deprecated in C++17)(removed in C++20) const T*
  typedef T& reference; // (deprecated in C++17)(removed in C++20) T&
  typedef const T& const_reference; // (deprecated in C++17)(removed in C++20) const T&

  template <class U>
  struct rebind
  {
    typedef managed_allocator<U> other;
  };
  managed_allocator() = default;
  template <class U>
  constexpr managed_allocator(const managed_allocator<U>&) noexcept
  {}
  T* allocate(std::size_t n)
  {
    void* out = nullptr;
    check(cudaMallocManaged(&out, n * sizeof(T)));
    return static_cast<T*>(out);
  }
  void deallocate(T* p, std::size_t) noexcept
  {
    check(cudaFree(p));
  }
};
template <class T, class... Args>
T* make_(Args&&... args)
{
  managed_allocator<T> ma;
  return new (ma.allocate(1)) T(std::forward<Args>(args)...);
}

template <class String>
void do_trie(String const& input, bool use_cuda, int blocks, int threads)
{
  std::vector<trie, managed_allocator<trie>> nodes(1 << 17);
  if (use_cuda)
  {
    check(cudaMemset(nodes.data(), 0, nodes.size() * sizeof(trie)));
  }

  auto t = nodes.data();
  auto b = make_<cuda::std::atomic<trie*>>(nodes.data() + 1);

  auto const begin = std::chrono::steady_clock::now();
  std::atomic_signal_fence(std::memory_order_seq_cst);
  if (use_cuda)
  {
    call_make_trie<<<blocks, threads>>>(t, b, input.data(), input.data() + input.size());
    check(cudaDeviceSynchronize());
  }
  else
  {
    assert(blocks == 1);
    std::vector<std::thread> tv(threads);
    for (auto count = threads; count; --count)
    {
      tv[count - 1] = std::thread([&, count]() {
        make_trie(*t, *b, input.data(), input.data() + input.size(), count - 1, threads);
      });
    }
    for (auto& t : tv)
    {
      t.join();
    }
  }
  std::atomic_signal_fence(std::memory_order_seq_cst);
  auto const end   = std::chrono::steady_clock::now();
  auto const time  = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
  auto const count = b->load() - nodes.data();
  std::cout << "Assembled " << count << " nodes on " << blocks << "x" << threads << " " << (use_cuda ? "cuda" : "cpu")
            << " threads in " << time << "ms." << std::endl;
}

int main()
{
  std::basic_string<char, std::char_traits<char>, managed_allocator<char>> input;

  char const* files[] = {
    "books/2600-0.txt",
    "books/2701-0.txt",
    "books/35-0.txt",
    "books/84-0.txt",
    "books/8800.txt",
    "books/pg1727.txt",
    "books/pg55.txt",
    "books/pg6130.txt",
    "books/pg996.txt",
    "books/1342-0.txt"};

  for (auto* ptr : files)
  {
    std::cout << ptr << std::endl;
    auto const cur = input.size();
    std::ifstream in(ptr);
    if (in.fail())
    {
      std::cerr << "Failed to open file: " << ptr << std::endl;
      return -1;
    }
    in.seekg(0, std::ios_base::end);
    auto const pos = in.tellg();
    input.resize(cur + pos);
    in.seekg(0, std::ios_base::beg);
    in.read((char*) input.data() + cur, pos);
  }

  do_trie(input, false, 1, 1);
  do_trie(input, false, 1, 1);
  do_trie(input, false, 1, std::thread::hardware_concurrency());
  do_trie(input, false, 1, std::thread::hardware_concurrency());

  assert(cudaSuccess == cudaSetDevice(0));
  cudaDeviceProp deviceProp;
  assert(cudaSuccess == cudaGetDeviceProperties(&deviceProp, 0));

  do_trie(input, true, deviceProp.multiProcessorCount * deviceProp.maxThreadsPerMultiProcessor >> 10, 1 << 10);
  do_trie(input, true, deviceProp.multiProcessorCount * deviceProp.maxThreadsPerMultiProcessor >> 10, 1 << 10);

  return 0;
}
