.. _cccl-runtime-cudart-interactions:

CUDA Runtime interactions
=========================

Some runtime objects have a non-owning ``_ref`` counterpart (for example, ``stream`` and ``stream_ref``). Prefer the
owning type for lifetime management, and use the ``_ref`` type for code that would otherwise accept a C++ reference but
needs to interoperate with existing CUDA Runtime code.

CCCL runtime types that wrap CUDA Runtime handles support interoperating with CUDA Runtime handles via ``get()``,
constructors that accept native handles, ``release()``, and ``from_native_handle`` helpers. This makes it straightforward
to bridge between cccl-runtime APIs and existing CUDA Runtime code without losing ownership clarity.

Use ``get()`` on both owning and non-owning types. Constructors from native handles are intended for ``_ref`` wrappers,
while ``release()`` and ``from_native_handle`` are for owning objects that transfer or assume ownership.

Example: handle interop patterns
--------------------------------

.. code:: cpp

   #include <cuda/stream>

   void use_handle_interop(cuda::device_ref device, cudaStream_t raw_stream) {
     // _ref from native handle (non-owning).
     cuda::stream_ref borrowed{raw_stream};

     // Universal handle access.
     assert(borrowed.get() == raw_stream);

     // Owning from native handle (assumes ownership).
     auto owned = cuda::stream::from_native_handle(raw_stream);

     assert(owned.get() == raw_stream);

     // Release ownership back to CUDA Runtime.
     cudaStream_t released = owned.release();

     assert(released == raw_stream);
   }

Device selection
----------------

The Runtime API emphasizes explicit device selection. Most entry points take a ``cuda::device_ref`` or a device-bound
resource (such as ``cuda::stream{device}``) rather than relying on implicit global state like ``cudaSetDevice``. This
makes device ownership and lifetime clearer, especially in multi-GPU code.

The current device can still be set via the CUDA Runtime, but cccl-runtime APIs ignore that global state and require an
explicit device argument. cccl-runtime also does not provide APIs that read or mutate the current device, by design.


Default stream interop
----------------------

The CUDA default (NULL) stream is not exposed as a first-class runtime object because it is tied to implicit per-device
state and encourages hidden dependencies. Instead, it can be wrapped into ``cuda::stream_ref`` when needed for interop.

.. note::

   When wrapping the NULL stream, the current device must be set explicitly first. CUDA binds the NULL stream to the
   active device, so the wrapper must be created after selecting the correct device.

Example: wrapping the default stream
------------------------------------

.. code:: cpp

   #include <cuda/stream>

   void use_default_stream(int device_id) {
     cudaSetDevice(device_id);

     cuda::stream_ref default_stream{cudaStreamPerThread};
     // Use default_stream with cccl-runtime APIs.
   }

The above applies to Driver API interop cases as well, where the current context must be managed by the user rather than
the current device setting.
