.. _libcudacxx-ptx-examples-device-tensormap-initialization:

How to initialize a tensor map on device
========================================

An introduction to TMA (Tensor Memory Access) can be found in the `CUDA
Programming Guide
<https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-data-copies-using-tensor-memory-access-tma>`__.
It describes how to create a tensor map on the **host** using the CUDA driver
API.

This example explains how to initialize a tensor map on **device**. This is
useful in situations where the typical way of transferring the tensor map (using
``const __grid_constant__`` kernel parameters) is undesirable, for instance,
when processing a batch of tensors of various sizes in a single kernel launch.

The high-level structure is as follows:

1. Create a tensor map "template" using the Driver API on the host.
2. Modify the template in a kernel on device and store the initialized tensor
   map in device memory.
3. Use the tensor map in a kernel with appropriate fencing.

The high-level code structure is displayed below.

.. literalinclude:: ./device_tensormap_initialization.cu
    :language: c++
    :dedent:
    :start-after: example-begin overview
    :end-before: example-end overview

The sections below describe the high-level steps. The use of the driver API is
described at the end for completeness, as it is already described in the CUDA
Programming Guide.

Throughout the examples, the following ``tensormap_params`` struct contains the
new values of the fields to be updated. It is included here, to reference when
reading the examples in the sections below.

.. literalinclude:: ./device_tensormap_initialization.cu
    :language: c++
    :dedent:
    :start-after: example-begin tensormap_params
    :end-before: example-end tensormap_params

Device-side initialization and modification of a tensor map
-----------------------------------------------------------

The process of initializing a tensormap in global memory proceeds as follows.

1. Pass an existing tensor map, the template, to the kernel. In contrast to
   kernels that ***use** the tensor map in a ``cp.async.bulk.tensor`` instruction,
   this may be done in any way: a pointer to global memory, kernel parameter, a
   ``__const___`` variable, etc.

2. Copy the template tensor map to shared memory.

3. Modify the tensor map in shared memory using :ref:`tensormap.replace
   <libcudacxx-ptx-instructions-tensormap-replace>`. This instruction can be
   used to modify any field of the tensor map, including the base addres, size,
   stride, etc.

4. Copy the tensor map from shared memory to global memory using the
   :ref:`tensormap.cp_fenceproxy
   <libcudacxx-ptx-instructions-tensormap-cp_fenceproxy>` instruction.

The code below contains a kernel that follows these steps. For completeness, it
modifies all the fields of the tensor map. Typically, a kernel will modify just
a few fields.

In this kernel, the tensor map template is passed as a kernel parameter. This is
the preferred way of moving the template from the host to device. If the kernel
is intended to update an existing tensor map in device memory, it can take a
pointer to the existing tensor map to modify.

**Note**: The format of the tensor map may change from one GPU architecture to
the next. Therefore, the ``tensormap.replace`` instructions are marked as specific
to ``sm_90a``. To use them, compile using ``nvcc -arch sm_90a ...``.

.. literalinclude:: ./device_tensormap_initialization.cu
    :language: c++
    :dedent:
    :start-after: example-begin modification
    :end-before: example-end modification

Usage of a modified tensor map
------------------------------

In contrast to using a tensor map that is passed as a ``const __grid_constant__``
kernel parameter, using a tensor map in global memory requires establishing a
release-acquire pattern in the tensor map proxy between the threads that modify
the tensor map and the threads that use it.

The release part of the pattern was shown in the previous section. It is
accomplished using the :ref:`tensormap.cp_fenceproxy
<libcudacxx-ptx-instructions-tensormap-cp_fenceproxy>` instruction.

The acquire part is accomplished using the :ref:`fence.proxy.tensormap::generic
<libcudacxx-ptx-instructions-fence>` instruction. If the two threads are on the
same device, the scope ``.gpu`` suffices. If threads are on different devices, the
``.sys`` scope must be used. Once a tensormap has been acquired, it can be used by
threads in the block, after sufficient synchronization, e.g., using
``__syncthreads()``. The thread that uses the tensormap and the thread that
performs the fence must be in the same block. The fence **cannot be delegated**
to a thread in another block or kernel launch. If there are not intermediate
modifications, the fence does not have be repeated before each
``cp.async.bulk.tensor`` instruction.

The ``fence`` and subsequent use of the tensor map is shown in the example below.

.. literalinclude:: ./device_tensormap_initialization.cu
    :language: c++
    :dedent:
    :start-after: example-begin use
    :end-before: example-end use


Creating a template tensor map using the driver API
---------------------------------------------------

The following code creates a minimal tensor map that can be further modified on
device.

.. literalinclude:: ./device_tensormap_initialization.cu
    :language: c++
    :dedent:
    :start-after: example-begin make-template
    :end-before: example-end make-template

The ``get_cuTensorMapEncodeTiled`` function is omitted from the code example. It
can be found in the CUDA Programming Guide or in the full code example linked
below.

Full code example
-----------------

The full code example is included below. The `cuda::ptx` instructions in this
tutorial have become available in CUDA Toolkit 12.5.

.. literalinclude:: ./device_tensormap_initialization.cu
    :language: c++
