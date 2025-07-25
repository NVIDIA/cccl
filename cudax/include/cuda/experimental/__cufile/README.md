# CUDA I/O Library - Modern C++ Bindings for cuFILE

Modern C++ bindings for NVIDIA cuFILE (GPU Direct Storage) providing direct API mapping with RAII resource management.

## Design Principles

- **Direct API Mapping**: C++ functions map directly to cuFILE C API
- **RAII Management**: Automatic resource cleanup for files, buffers, batches, and streams
- **Modern C++**: Exception-safe, move-only semantics
- **Type Safety**: Strong typing with minimal overhead

## Core Components

### File Operations
```cpp
auto file = cuda::io::file_handle{"data.bin", std::ios_base::in};
size_t bytes = file.read(gpu_buffer, size, file_offset, buffer_offset);
file.read_async(gpu_buffer, &size, &file_offset, &buffer_offset, &bytes_read, stream);
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
    {gpu_buffer, file_offset, buffer_offset, size, CUFILE_READ, cookie}
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
#include <cuda/io/cufile.hpp>

// Check availability
if (!cuda::io::is_cufile_available()) return 1;

// Allocate and register GPU buffer
void* gpu_buffer;
cudaMalloc(&gpu_buffer, size);
auto buffer = cuda::io::buffer_handle{gpu_buffer, size};

// File operations
auto file = cuda::io::file_handle{"data.bin", std::ios_base::in};
size_t bytes_read = file.read(gpu_buffer, size);

cudaFree(gpu_buffer);
```

### Asynchronous Operations
```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);
auto stream_handle = cuda::io::stream_handle{stream};

size_t size = 1024 * 1024;
off_t file_offset = 0, buffer_offset = 0;
ssize_t bytes_read;

file.read_async(gpu_buffer, &size, &file_offset, &buffer_offset, 
                &bytes_read, stream);
cudaStreamSynchronize(stream);
```

### Batch Operations
```cpp
auto batch = cuda::io::batch_handle{10};
std::vector<cuda::io::batch_io_params> ops = {
    {buffer1, 0, 0, size1, CUFILE_READ, nullptr},
    {buffer2, size1, 0, size2, CUFILE_READ, nullptr}
};

batch.submit(file, ops);
auto results = batch.get_status(ops.size());
```

## Key Features

- **Complete API Coverage**: All cuFILE functionality exposed
- **Exception Safety**: RAII ensures no resource leaks
- **Performance**: Direct API mapping with minimal overhead
- **Compatibility**: Works with existing CUDA code 