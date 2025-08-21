CUDA cuFILE (GPU Direct Storage) — experimental C++ bindings
===========================================================

Overview
--------
``cuda::experimental::cufile`` provides modern C++ (RAII) wrappers over the NVIDIA cuFILE C API
for GPU Direct Storage (GDS). It offers file, buffer, stream, and batch I/O abstractions that map
directly to the underlying cuFILE operations.

Availability and feature guard
------------------------------
This feature depends on the cuFILE SDK. The public header defines the feature macro
``CUDAX_HAS_CUFILE`` based on the availability of the C header ``<cufile.h>``:

.. code-block:: c++

  #include <cuda/experimental/cufile.h>

  #if CUDAX_HAS_CUFILE()
  // cuFILE is available on this system
  #endif

If cuFILE is not available, the header remains includable but provides no APIs; use
``#if CUDAX_HAS_CUFILE`` to conditionally compile cuFILE-dependent code.

Prerequisites
-------------
- NVIDIA GPU supported by CUDA
- CUDA Toolkit
- GPU Direct Storage/cuFILE installed (``<cufile.h>`` available to the compiler)

Quick start
-----------
.. code-block:: c++

  #include <cuda/experimental/cufile.h>
  #include <cuda_runtime.h>

  int main() {
    cuda::experimental::cufile::driver_handle driver; // RAII: open/close cuFILE driver

    void* dev_ptr{};
    size_t n = 1 << 20;
    cudaMalloc(&dev_ptr, n);

    auto fh = cuda::experimental::cufile::file_handle{"data.bin", std::ios_base::out};
    cuda::std::span<const char> buf{static_cast<const char*>(dev_ptr), n};
    fh.write(buf);

    cudaFree(dev_ptr);
    return 0;
  }

Notes
-----
- For best performance and compatibility, align buffers and offsets to 4 KiB.
- Use ``file_handle`` for owning descriptors or ``file_handle_ref`` to wrap an existing ``fd``.
- Async APIs accept ``cuda::stream_ref`` and integrate with CUDA streams.



