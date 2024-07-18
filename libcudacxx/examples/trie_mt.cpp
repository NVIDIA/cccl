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

#include <atomic>
#include <cstddef>
#include <cstdint>

template <class T>
static constexpr T min(T a, T b)
{
  return a < b ? a : b;
}

struct trie
{
  struct ref
  {
    std::atomic<trie*> ptr = LIBCUDACXX_ATOMIC_VAR_INIT(nullptr);
    // the flag will protect against multiple pointer updates
    std::atomic_flag flag = LIBCUDACXX_ATOMIC_VAR_INIT(0);
  } next[26];
  std::atomic<int> count = LIBCUDACXX_ATOMIC_VAR_INIT(0);
};
int index_of(char c)
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
void make_trie(/* trie to insert word counts into */ trie& root,
               /* bump allocator to get new nodes*/ std::atomic<trie*>& bump,
               /* input */ const char* begin,
               const char* end,
               /* thread this invocation is for */ unsigned index,
               /* how many threads there are */ unsigned domain)
{
  auto const size   = end - begin;
  auto const stride = (size / domain + 1);

  auto off        = min(size, stride * index);
  auto const last = min(size, off + stride);

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
        n->count.fetch_add(1, std::memory_order_relaxed);
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
    if (n->next[index].ptr.load(std::memory_order_acquire) == nullptr)
    {
      if (n->next[index].flag.test_and_set(std::memory_order_relaxed) == 1)
      {
        while (n->next[index].ptr.load(std::memory_order_acquire) == nullptr)
          ;
      }
      else
      {
        auto next = bump.fetch_add(1, std::memory_order_relaxed);
        n->next[index].ptr.store(next, std::memory_order_release);
      }
    }
    n = n->next[index].ptr.load(std::memory_order_relaxed);
  }
}

#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

void do_trie(std::string const& input, int threads)
{
  std::vector<trie> nodes(1 << 17);

  auto t = nodes.data();
  std::atomic<trie*> b(nodes.data() + 1);

  auto const begin = std::chrono::steady_clock::now();
  std::atomic_signal_fence(std::memory_order_seq_cst);
  std::vector<std::thread> tv(threads);
  for (auto count = threads; count; --count)
  {
    tv[count - 1] = std::thread([&, count]() {
      make_trie(*t, b, input.data(), input.data() + input.size(), count - 1, threads);
    });
  }
  for (auto& t : tv)
  {
    t.join();
  }
  std::atomic_signal_fence(std::memory_order_seq_cst);
  auto const end   = std::chrono::steady_clock::now();
  auto const time  = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
  auto const count = b.load() - nodes.data();
  std::cout << "Assembled " << count << " nodes on " << threads << " cpu threads in " << time << "ms." << std::endl;
}

int main()
{
  std::string input;

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

  do_trie(input, 1);
  do_trie(input, 1);
  do_trie(input, std::thread::hardware_concurrency());
  do_trie(input, std::thread::hardware_concurrency());

  return 0;
}
