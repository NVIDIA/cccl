#include <thrust/fill.h>
#include <thrust/mr/new.h>

#include <unittest/unittest.h>

template <typename MemoryResource>
void test_alignment(MemoryResource memres, std::size_t size, std::size_t alignment)
{
  void* ptr = memres.do_allocate(size, alignment);
  REQUIRE(reinterpret_cast<std::size_t>(ptr) % alignment == 0u);

  char* char_ptr = reinterpret_cast<char*>(ptr);
  thrust::fill(char_ptr, char_ptr + size, char{});

  memres.do_deallocate(ptr, size, alignment);
}

static const std::size_t MinTestedSize  = 32;
static const std::size_t MaxTestedSize  = 8 * 1024;
static const std::size_t TestedSizeStep = 1;

static const std::size_t MinTestedAlignment   = 16;
static const std::size_t MaxTestedAlignment   = 4 * 1024;
static const std::size_t TestedAlignmentShift = 1;

TEST_CASE("TestNewDeleteResourceAlignedAllocation", "[mr_new]")
{
  for (std::size_t size = MinTestedSize; size <= MaxTestedSize; size += TestedSizeStep)
  {
    for (std::size_t alignment = MinTestedAlignment; alignment <= MaxTestedAlignment;
         alignment <<= TestedAlignmentShift)
    {
      test_alignment(thrust::mr::new_delete_resource(), size, alignment);
    }
  }
}
