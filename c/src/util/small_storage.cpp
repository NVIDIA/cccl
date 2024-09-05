#include <cstring>
#include <memory>

// copy_or_allocate_into_aligned_storage is a helper that will allocate space to hold an object and some aligned data
// after it in one allocation. A return value of 0 indicates that the buffer pointed to by 'space' was sufficient for
// holding the additional data. a non-zero return means a buffer was allocated large enough to include the
// minimum size as determined by the offset.
void* copy_or_allocate_into_aligned_storage(
  void* space, // Pointer to buffer in which to copy memory
  size_t space_size, // Space available in this buffer (number of bytes available after offset)
  size_t offset, // Offset into memory from which to begin alignment (space+offset == beginning of buffer)
  void* source, // Source buffer to copy from
  size_t src_alignment, // Source's required alignment,
  size_t src_size // Source's size
)
{
  char* beginning_space_addr = (char*) space;
  void* buffer_addr          = beginning_space_addr + offset;
  void* aligned_addr         = std::align(src_alignment, src_size, buffer_addr, space_size);
  void* allocation           = nullptr;

  if (!aligned_addr)
  {
    // Allocate enough space to fit the entirety of the buffer, this may overallocate by at most
    // src_alignment - offset
    size_t allocation_size = offset + src_alignment + src_size;

    allocation           = malloc(allocation_size);
    beginning_space_addr = (char*) allocation;
    buffer_addr          = beginning_space_addr + offset;
    aligned_addr         = std::align(src_alignment, src_size, buffer_addr, allocation_size);
  }

  memcpy(aligned_addr, source, src_size);
  return allocation;
}
