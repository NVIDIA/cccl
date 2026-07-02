.. _cccl-development-visibility-device-kernel-visibility:


Device Kernel Visibility Issue
-------------------------------

Consider the following simple translation unit (TU):

.. code-block:: cpp

    template <class T>
    __global__ void kernel(T *val) {
        ::printf("kernel: set val = 42\n");
        *val = 42;
    }

   int main() {
     int *ptr{};
     kernel<<<1, 1>>>(ptr);
   }

The cuda compiler frontend will turn this into:

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

However, imagine that there are two shared libraries: ``lib_a`` and ``lib_b`` both instantiating different ``kernel``
instances, e.g ``d_kernel<int>`` and ``d_kernel<size_t>``.

.. code-block:: cmake

    project(DeviceKernelVisibility CUDA CXX)

    add_executable(device_kernel_visibility main.cu)
    add_library(lib_a SHARED tu_a.cu)
    add_library(lib_b SHARED tu_b.cu)
    target_link_libraries(device_kernel_visibility PRIVATE lib_a lib_b)

Each library will have it's own fatbinary: ``d_kernel<int>_a`` and ``d_kernel<size_t>_b`` as well as host stub functions
``h_kernel<int>_a`` and ``h_kernel<size_t>_b``.

=== ============= ============
lib host          device
=== ============= ============
a   0xh_kernel_a  0xd_kernel_a
b   0xh_kernel_b  0xd_kernel_b
=== ============= ============

In contrast to
:ref:`Problem 1 <cccl-development-visibility-host-stub-visibility>` the host stubs will get a different mangled name
and so the right stub function will always be selected.

Now imagine that both libraries are going to defer launching of their kernels to a function ``foo`` common to both
``lib_a`` and ``lib_b``, which has weak external linkage. This might happen in ``CUB``, because it launches
kernels through the ``thrust::triple_chevron`` helper.

Similar to :ref:`Problem 1 <cccl-development-visibility-host-stub-visibility>` the linker will pick one of the two
weak symbols and subsequently ``lib_a`` will try to pass its own kernel ``d_kernel<int>_a`` to ``lib_b::foo``.

However, the CUDA runtime in ``lib_b`` will not find any kernel registered at the address of ``d_kernel<int>_a`` and
will fail to launch the kernel.

A simple example program that exemplifies this can be found
`on github <https://github.com/NVIDIA/cccl/tree/main/docs/cub/developer/visibility/examples/device_kernel_visibility>`_

.. code-block:: bash

   ./device_kernel_visibility/device_kernel_visibility
   a: kernel stub address: 0x7fdec19e13eb                      <== launching kernel_a from a
   a: kernel is in mapping: no error
   b: launched kernel
   a: kernel: set val = 42
   a: synchronized stream
   a: copied from device to host
   a: out: 42
   a: kernel was launched: out == 42                           <== everything is fine

   a: defers launch to b
   b: kernel stub address: 0x7fdec19e13eb                      <== launch kernel_a from b
   b: kernel NOT found in mapping: invalid device function     <== kernel_a is not found in b mapping
   b: FAILED to launch kernel                                  <== unable to launch the kernel from b
   b: synchronized stream
   b: copied from device to host
   b: out: 0
   b: kernel was NOT actually launched: out != 42

   b: kernel stub address: 0x7fdec19333eb                      <== launch kernel_b from b
   b: kernel is in mapping: no error
   b: launched kernel
   b: kernel: set val = 42
   b: synchronized stream
   b: copied from device to host
   b: out: 42
   b: kernel was launched: out == 42                           <== everything is fine

   b: defers launch to a
   a: kernel stub address: 0x7fdec19333eb                      <== launching kernel_b from a
   a: kernel NOT found in mapping: invalid device function     <== same issue as above
   a: FAILED to launch kernel
   b: kernel: set val = 42
   a: synchronized stream
   a: copied from device to host
   a: out: 42
   a: kernel was launched: out == 42                           <==  kernel launch somehow succeeded
