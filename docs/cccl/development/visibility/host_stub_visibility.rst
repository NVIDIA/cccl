.. _cccl-development-visibility-host-stub-visibility:


Host Stub Visibility Issue
---------------------------

Consider the following simple translation unit (TU):

.. code-block:: cpp

  #include <cstdio>
  #include <cuda/memory>

  template <class T>
  __global__ void kernel(T *val) {
      printf("kernel: set val = 42\n");
      *val = 42;
  }

  __device__ int val;

  int main() {

     kernel<<<1, 1>>>(cuda::get_device_address(val));
  }

The CUDA compiler frontend will turn this into:

.. code-block:: cpp

   template< class T>
   static void __wrapper__device_stub_kernel(T *&ptr) {
     ::cudaLaunchKernel(0, 0, 0, 0, 0, 0);
   }

   // stub host function
   template< class T>
   void kernel(T *ptr) {
     __wrapper__device_stub_kernel<T>(ptr);
   }

   int main() {
     int *ptr{};
     (__cudaPushCallConfiguration(1, 1)) ? (void)0 : kernel(ptr);
   }

   static void __device_stub__Z6kernelIiEvPT_(int *__par0) {
     __cudaLaunchPrologue(1);
     __cudaSetupArgSimple(__par0, 0UL);
     __cudaLaunch(((char *)((void ( *)(int *))kernel )));
   }

   template<> void __wrapper__device_stub_kernel(int *&__cuda_0) {
     __device_stub__Z6kernelIiEvPT_( (int *&)__cuda_0);
   }

The CUDA runtime is going to use the address of ``template<> void kernel(T *ptr)`` (in the following ``h_kernel``)
as a key in the host stub function (``h_kernel``) - device function (``d_kernel``) mapping. This works fine if
there is only a single source of truth for the stub function ``h_kernel``.

However, imagine that there are two shared libraries: ``lib_a`` and ``lib_b`` both using the same ``kernel`` instance.

.. code-block:: cmake

    project(HostStubVisibility CUDA CXX)

    add_executable(host_stub_visibility main.cu)
    add_library(lib_a SHARED tu_a.cu)
    add_library(lib_b SHARED tu_b.cu)
    target_link_libraries(host_stub_visibility PRIVATE lib_a lib_b)

Each library will have it's own fatbinary: ``d_kernel_a`` and ``d_kernel_b``, but the the compiler
generated host stub function ``h_kernel`` has weak external linkage, so after dynamic linkage, we'll end up having
only one of them.

=== ===================== ============
lib host                  device
=== ===================== ============
a   0xh_kernel_a          0xd_kernel_a
b   0xh_kernel_a <- issue 0xd_kernel_b
=== ===================== ============

Since there's a clash of stub function addresses, only one entry stored. When ``lib_b`` queries for the
kernel using its address of ``h_kernel``, it's visible, although it might point to ``lib_a``'s fatbinary.
The opposite case might happen as well, depending on loading order, linker etc and is undefined behavior.

Launching ``d_kernel`` from ``lib_b`` is not possible and leads to random errors. For instance, there seems to be
some per CUDART global state. When the ``__cudaPushCallConfiguration`` is called in ``lib_b``, it affects the state of
``cudart_b``, but the launch happens through ``h_kernel``, which is in ``lib_a``.

This sometimes leads to ``__global__ function call is not configured``. However, there might also be no error at all,
and the kernel launch is silently skipped.

A simple example program that exemplifies this can be found
`on github <https://github.com/NVIDIA/cccl/tree/main/docs/cub/developer/visibility/examples/host_stub_visibility>`_

.. code-block:: bash

   :./host_stub_visibility/host_stub_visibility
   a: kernel stub address: 0x7f43318a415d           <== same address as in B
   a: kernel is in mapping: no error                <== kernel is found in the mapping
   b: launched kernel
   a: kernel: set val = 42
   a: synchronized stream
   a: copied from device to host
   a: out: 42
   a: kernel was launched: out == 42

   b: kernel stub address: 0x7f43318a415d           <== same address as in A
   b: kernel is in mapping: no error                <== kernel is found in the mapping
   b: launched kernel
   b: synchronized stream
   b: copied from device to host
   b: out: 0
   b: kernel was NOT actually launched: out != 42   <== silent failure
