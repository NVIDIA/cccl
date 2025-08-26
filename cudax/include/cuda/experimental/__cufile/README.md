# CUDA I/O Library - Modern C++ Bindings for cuFILE

Modern C++ bindings for NVIDIA cuFILE (GPU Direct Storage) providing direct API mapping with RAII resource management.

## Design Principles

- **Direct API Mapping**: C++ functions map directly to cuFILE C API
- **RAII Management**: Automatic resource cleanup for files, buffers, batches, and streams
- **Modern C++**: Exception-safe, move-only semantics
- **Type Safety**: Strong typing with minimal overhead

## Core Components

### File Handle Types

The library provides two file handle types with different ownership semantics:

- **`file_handle`** - Owning RAII handle that closes the file descriptor on destruction
- **`file_handle_ref`** - Non-owning reference that doesn't close the file descriptor

Both types inherit from `file_handle_base` and provide the same I/O operations.

### File Operations
```cpp
// Owning file handle - closes file descriptor on destruction
auto file = cuda::experimental::cufile::file_handle{"data.bin", std::ios_base::in};
size_t bytes = file.read(gpu_buffer, file_offset, buffer_offset);
file.read_async(stream, gpu_buffer, file_offset, buffer_offset);

// Non-owning file handle reference - doesn't close file descriptor
int fd = open("data.bin", O_RDONLY | O_DIRECT);
auto file_ref = cuda::experimental::cufile::file_handle_ref{fd};
size_t bytes = file_ref.read(gpu_buffer, file_offset, buffer_offset);
// fd must be closed manually when done
close(fd);
```

### Buffer Registration
```cpp
auto buffer = cuda::io::buffer_handle{gpu_ptr, size, flags};
// Automatically deregistered on destruction
```

### Batch Operations
```cpp
auto batch = cuda::io::batch_handle{max_operations};
std::vector<cuda::io::batch_io_params> operations = {
    {gpu_buffer, file_offset, buffer_offset, size, cuda::experimental::cufile::cu_file_opcode::read, cookie}
};
batch.submit(file, operations);
auto results = batch.get_status(min_completed, timeout_ms);
```

### Stream Management
```cpp
auto stream_handle = cuda::io::stream_handle{cuda_stream, flags};
// Automatically deregistered on destruction
```

### Driver Management
```cpp
cuda::io::driver_open();
cuda::io::set_poll_mode(true, 4);
auto props = cuda::io::get_driver_properties();
cuda::io::driver_close();
```

## Usage Examples

### Basic Operations
```cpp
#include <cuda/experimental/cufile.h>

// Allocate GPU buffer
void* gpu_buffer;
cudaMalloc(&gpu_buffer, size);

// Owning file handle - automatically closes file descriptor
auto file = cuda::experimental::cufile::file_handle{"data.bin", std::ios_base::in};
cuda::std::span<char> buffer_span{static_cast<char*>(gpu_buffer), size};
size_t bytes_read = file.read(buffer_span);

// Non-owning file handle reference - for existing file descriptors
int fd = open("another_file.bin", O_RDONLY | O_DIRECT);
auto file_ref = cuda::experimental::cufile::file_handle_ref{fd};
size_t bytes_read2 = file_ref.read(buffer_span);
close(fd); // Manual cleanup required

cudaFree(gpu_buffer);
```

### Asynchronous Operations
```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);
cuda::stream_ref stream_ref{stream};

// Using owning file handle
auto file = cuda::experimental::cufile::file_handle{"data.bin", std::ios_base::in};
cuda::std::span<char> buffer_span{static_cast<char*>(gpu_buffer), size};
file.read_async(stream_ref, buffer_span, file_offset, buffer_offset);
cudaStreamSynchronize(stream);

// Using non-owning file handle reference
int fd = open("data.bin", O_RDONLY | O_DIRECT);
auto file_ref = cuda::experimental::cufile::file_handle_ref{fd};
file_ref.write_async(stream_ref, buffer_span, file_offset, buffer_offset);
cudaStreamSynchronize(stream);
close(fd);
```

### Batch Operations
```cpp
// Works with both file_handle and file_handle_ref
auto file = cuda::experimental::cufile::file_handle{"data.bin", std::ios_base::in};

auto batch = cuda::experimental::cufile::batch_handle{10};
std::vector<cuda::experimental::cufile::batch_io_params_span<char>> ops = {
    {buffer_span1, 0, 0, cuda::experimental::cufile::cu_file_opcode::read},
    {buffer_span2, size1, 0, cuda::experimental::cufile::cu_file_opcode::read}
};

batch.submit(file, ops);
auto results = batch.get_status(ops.size());

// Can also use file_handle_ref with batch operations
int fd = open("data.bin", O_RDONLY | O_DIRECT);
auto file_ref = cuda::experimental::cufile::file_handle_ref{fd};
batch.submit(file_ref, ops);
close(fd);
```

## Key Features

- **Complete API Coverage**: All cuFILE functionality exposed
- **Exception Safety**: RAII ensures no resource leaks
- **Performance**: Direct API mapping with minimal overhead
- **Compatibility**: Works with existing CUDA code
