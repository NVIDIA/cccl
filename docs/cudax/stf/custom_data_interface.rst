.. _stf_custom_data_interface:

CUDASTF offers an extensible API that allows users to implement their
own data interface.

Let us for example go through the different steps to implement a data
interface for a very simple simple implementation of a matrix class.

For the sake of simplicity, we here only consider the CUDA stream
backend, but adding support for the CUDA graph backend simply require
some extra steps which use the CUDA graph API.

Implementation of the ``matrix`` class
======================================

For the sake of simplicity, we consider a very simple representation of
matrix, only defined by the dimensions m and n, and by the base address
of the matrix which we assume to be contiguous.

.. code:: c++

   template <typename T>
   class matrix {
   public:
       matrix(size_t m, size_t n, T* base) : m(m), n(n), base(base) {}
       __host__ __device__ T& operator()(size_t i, size_t j) { return base[i + j * m]; }
       __host__ __device__ const T& operator()(size_t i, size_t j) const { return base[i + j * m]; }
       size_t m, n;
       T* base;
   };

Defining the shape of a matrix
==============================

The first step consists in defining what is the *shape* of a matrix. The
shape of a matrix should be a class that defines all parameters which
are the same for all data instances, ``m`` and ``n``. On the other hand,
the base address should not be part of this shape class, because each
data instance will have its own base address.

To define what is the shape of a matrix, we need to specialize the
``cudastf::shape_of`` trait class.

.. code:: c++

   template <typename T>
   class cudastf::shape_of<matrix<T>> {
   public:
       /**
        * @brief The default constructor.
        *
        * All `shape_of` specializations must define this constructor.
        */
       shape_of() = default;

       explicit shape_of(size_t m, size_t n) : m(m), n(n) {}

       /**
        * @name Copies a shape.
        *
        * All `shape_of` specializations must define this constructor.
        */
       shape_of(const shape_of&) = default;

       /**
        * @brief Extracts the shape from a matrix
        *
        * @param M matrix to get the shape from
        *
        * All `shape_of` specializations must define this constructor.
        */
       shape_of(const matrix<T>& M) : shape_of<matrix<T>>(M.m, M.n) {}

       /// Mandatory method : defined the total number of elements in the shape
       size_t size() const { return m * n; }

       size_t m;
       size_t n;
   };

We here see that ``shape_of<matrix<T>>`` contains two ``size_t`` fields
``m`` and ``n``.

In addition, we need to define a default constructor and a copy
constructors.

To implement the ``.shape()`` member of the ``logical_data`` class, we
need to define a constructor which takes a const reference to a matrix.

Finally, if the ``ctx.parallel_for`` construct is needed, we must define
a ``size_t size() const`` method which computes the total number of
elements in a shape.

Hash of a matrix
================

For internal needs, such as using (unordered) maps of data instances,
CUDASTF need to have specialized forms of the ``std::hash`` trait class.

The ``()`` operator of this class should compute a unique identifier
associated to the description of the data instance. This typically means
computing a hash of the matrix sizes, and of the base address. Note that
this hash *does not* depend on the actual content of the matrix.

In code snippet, we are using the ``cudastf::hash_combine`` helper which
updates a hash value with another value. This function is available from
the ``cudastf/utility/hash.h`` header.

.. code:: c++

   template <typename T>
   struct std::hash<matrix<T>> {
       std::size_t operator()(matrix<T> const& m) const noexcept {
           // Combine hashes from the base address and sizes
           return cudastf::hash_all(m.m, m.n, m.base);
       }
   };

Defining a data interface
=========================

We can now implement the actual data interface for a matrix class, which
defines the basic operations that CUDASTF need to perform on a matrix.

The ``matrix_stream_interface`` class inherits from the
``data_interface`` class, but to implement a data interface using APIs
based on CUDA streams, ``matrix_stream_interface`` inherits from
``stream_data_interface_simple<matrix<T>>`` which contains pure virtual
functions that need to be implemented.

.. code:: c++

   template <typename T>
   class matrix_stream_interface : public stream_data_interface_simple<matrix<T>> {
   public:
       using base = stream_data_interface_simple<matrix<T>>;
       using base::shape_t;

       /// Initialize from an existing matrix
       matrix_stream_interface(matrix<T> m) : base(std::move(m)) {}

       /// Initialize from a shape of matrix
       matrix_stream_interface(shape_t s) : base(s) {}

       /// Copy the content of an instance to another instance
       ///
       /// This implementation assumes that we have registered memory if one of the data place is the host
       void stream_data_copy(const data_place& dst_memory_node, instance_id_t dst_instance_id,
               const data_place& src_memory_node, instance_id_t src_instance_id, cudaStream_t stream) override {
           assert(src_memory_node != dst_memory_node);

           cudaMemcpyKind kind = cudaMemcpyDeviceToDevice;
           if (src_memory_node == data_place::host) {
               kind = cudaMemcpyHostToDevice;
           }

           if (dst_memory_node == data_place::host) {
               kind = cudaMemcpyDeviceToHost;
           }

           const matrix<T>& src_instance = this->instance(src_instance_id);
           const matrix<T>& dst_instance = this->instance(dst_instance_id);

           size_t sz = src_instance.m * src_instance.n * sizeof(T);

           cuda_safe_call(cudaMemcpyAsync((void*) dst_instance.base, (void*) src_instance.base, sz, kind, stream));
       }

       /// allocate an instance on a specific data place
       ///
       /// setting *s to a negative value informs CUDASTF that the allocation
       /// failed, and that a memory reclaiming mechanism need to be performed.
       void stream_data_allocate(backend_ctx_untyped& ctx, const data_place& memory_node, instance_id_t instance_id, ssize_t& s,
               void** extra_args, cudaStream_t stream) override {
           matrix<T>& instance = this->instance(instance_id);
           size_t sz = instance.m * instance.n * sizeof(T);

           T* base_ptr;

           if (memory_node == data_place::host) {
               // Fallback to a synchronous method as there is no asynchronous host allocation API
               cuda_safe_call(cudaStreamSynchronize(stream));
               cuda_safe_call(cudaHostAlloc(&base_ptr, sz, cudaHostAllocMapped));
           } else {
               cuda_safe_call(cudaMallocAsync(&base_ptr, sz, stream));
           }

           // By filling a positive number, we notify that the allocation was successful
           *s = sz;

           instance.base = base_ptr;
       }

       /// deallocate an instance
       void stream_data_deallocate(backend_ctx_untyped& ctx, const data_place& memory_node, instance_id_t instance_id, void* extra_args,
               cudaStream_t stream) override {
           matrix<T>& instance = this->instance(instance_id);
           if (memory_node == data_place::host) {
               // Fallback to a synchronous method as there is no asynchronous host deallocation API
               cuda_safe_call(cudaStreamSynchronize(stream));
               cuda_safe_call(cudaFreeHost(instance.base));
           } else {
               cuda_safe_call(cudaFreeAsync(instance.base, stream));
           }
       }

       /// Register the host memory associated to an instance of matrix
       ///
       /// Note that this pin_host_memory method is not mandatory, but then it is
       /// the responsibility of the user to only passed memory that is already
       /// registered, and the allocation method on the host must allocate
       /// registered memory too. Otherwise, copy methods need to be synchronous.
       bool pin_host_memory(instance_id_t instance_id) override {
           matrix<T>& instance = this->instance(instance_id);
           if (!instance.base) {
               return false;
           }

           cuda_safe_call(pin_memory(instance.base, instance.m * instance.n * sizeof(T)));

           return true;
       }

       /// Unregister memory pinned by pin_host_memory
       void unpin_host_memory(instance_id_t instance_id) override {
           matrix<T>& instance = this->instance(instance_id);
           unpin_memory(instance.base);
       }
   };

``matrix_stream_interface`` must meet the following requirements so that
they can be used in the CUDA stream backend : - It must provide
constructors which take either a matrix, or a shape of matrix as
arguments. - It must implement the ``stream_data_copy``,
``stream_data_allocate`` and ``stream_data_deallocate`` virtual methods,
which respectively define how to copy an instance into another instance,
how to allocate an instance, and how to deallocate an instance. - It may
implement the ``pin_host_memory`` and ``unpin_host_memory`` virtual
methods which respectively register and unregister the memory associated
to an instance allocated on the host. These two methods are not
mandatory, but it is the responsibility of the user to either only pass
and allocate registered host buffers, or to ensure that the copy method
does not require such memory pinning. Similarly, accessing an instance
located in host memory from a device typically requires to access
registered memory.

Associating a data interface with the CUDA stream backend
=========================================================

To ensure that we can initialize a logical data from a matrix, or from
the shape of a matrix with ``stream_ctx::logical_data``, we then need to
specialize the ``cudastf::streamed_interface_of`` trait class.

The resulting class must simply define a type named ``type`` which is
the type of the data interface for the CUDA stream backend.

.. code:: c++

   template <typename T>
   class cudastf::streamed_interface_of<matrix<T>> {
   public:
       using type = matrix_stream_interface<T>;
   };

Once we have defined this trait class, it is for example possible to
initialize a logical data from a matrix, or from a matrix shape :

.. code:: c++

       std::vector<int> v(m * n, 0);
       matrix M(m, n, &v[0]);

       // Initialize from a matrix
       auto lM = ctx.logical_data(M);

       // Initialize from a shape
       auto lM2 = ctx.logical_data(shape_of<matrix<int>>(m, n));

Example of code using the ``matrix`` data interface
===================================================

We can now use the ``matrix`` class in CUDASTF, and access it from
tasks. In this code, we first initialize a matrix on the host, we then
apply a task which will update its content on the current device. We
finally check that the content is correct, by the means of the
write-back mechanism that automatically updates the reference data
instance of a logical data when calling ``ctx.sync()``.

.. code:: c++

   template <typename T>
   __global__ void kernel(matrix<T> M) {
       int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
       int nthreads_x = gridDim.x * blockDim.x;

       int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
       int nthreads_y = gridDim.y * blockDim.y;

       for (int x = tid_x; x < M.m; x += nthreads_x)
           for (int y = tid_y; y < M.n; y += nthreads_y) {
               M(x, y) += -x + 7 * y;
           }
   }

   int main() {
       stream_ctx ctx;

       const size_t m = 8;
       const size_t n = 10;
       std::vector<int> v(m * n);

       for (size_t j = 0; j < n; j++)
           for (size_t i = 0; i < m; i++) {
               v[i + j * m] = 17 * i + 23 * j;
           }

       matrix<int> M(m, n, &v[0]);

       auto lM = ctx.logical_data(M);

       // M(i,j) +=  -i + 7*i
       ctx.task(lM.rw())->*[](cudaStream_t s, auto dM) { kernel<<<dim3(8, 8), dim3(8, 8), 0, s>>>(dM); };

       ctx.sync();

       for (size_t j = 0; j < n; j++)
           for (size_t i = 0; i < m; i++) {
               assert(v[i + j * m] == (17 * i + 23 * j) + (-i + 7*i));
           }
   }
