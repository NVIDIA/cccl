.. _libcudacxx-extended-api-vector-types:

Vector Types
============

The ``<cuda/vector_types>`` header imports the vector types, such as ``float2`` and ``int3``, from the ``<vector_types.h>`` header that comes with the CUDA Toolkit to the ``cuda::`` namespace and provides several additional features to improve their usability.

.. rubric:: Factory Functions

The ``<vector_functions.h>`` header provides factory functions for all of the vector types. In C++, they are practically pointless because we can do `list initialization <https://en.cppreference.com/w/cpp/language/list_initialization>`__, but we provide them for consistency with the CUDA Toolkit and extend them to simplify the creation of the vector types.

For each type (in the example below, only ``float3`` is shown), the following factory functions are provided:
.. code:: cpp
   // equivalent to float3{}
   [[nodiscard]] __host__ __device__ inline constexpr
   float3 make_float3() noexcept;

   // equivalent to float3{x, y, z}
   [[nodiscard]] __host__ __device__ inline constexpr
   float3 make_float3(float x, float y, float z) noexcept;

   // disambiguation tag for the next overload
   inline constexpr value_broadcast_t value_broadcast{};

   // equivalent to float3{v, v, v}
   [[nodiscard]] __host__ __device__ inline constexpr
   float3 make_float3(value_broadcast_t /* tag */, float v) noexcept;

The ``cuda::value_broadcast_t`` type is a tag type that indicates that the value should be broadcasted to all components of the vector. This is useful for creating vectors with the same value in all components, such as creating a vector with all components set to the maximum value of the type. Don't construct the ``cuda::value_broadcast_t`` type directly, but use the ``cuda::value_broadcast`` variable instead.

.. rubric:: Generic Alternatives

To leverage the generic programming, the header provides generic alternatives to the vector types and the factory functions. The ``cuda::vector_type`` type provides is a type alias ``type`` to a vector type that matches the given type and size. The ``cuda::is_vector_type`` trait is an integral constant which can be used to determine if a given type is a *cv-qualified* CUDA vector type. The ``cuda::make_vector`` functions are generic equivalents to the factory functions described above.

.. code:: cpp
   template <class T, cuda::std::size_t Size>
   struct vector_type
   {
     using type = /* implementation-defined */;
   };

   template <class T, cuda::std::size_t Size>
   using vector_type_t = typename vector_type_t<T, Size>::type;

   template <class T>
   struct is_vector_type;

   template <class T>
   inline constexpr bool is_vector_type_v = is_vector_type<T>::value;

   template <class T, cuda::std::size_t Size>
   [[nodiscard]] __host__ __device__ inline constexpr
   vector_type_t<T, Size> make_vector() noexcept;

   template <class T, cuda::std::size_t Size, class... Args>
     requires (sizeof...(Args) == Size)
   [[nodiscard]] __host__ __device__ inline constexpr
   vector_type_t<T, Size> make_vector(Args... args) noexcept;

   template <class T, cuda::std::size_t Size>
   [[nodiscard]] __host__ __device__ inline constexpr
   vector_type_t<T, Size> make_vector(value_broadcast_t /* tag */, cuda::std::type_identity_t<T> v) noexcept;


.. rubric:: Tuple Protocol

The ``<cuda/vector_types>`` also implements the `tuple protocol <https://en.cppreference.com/w/cpp/utility/tuple/tuple-like>`__ for the vector types. This means that the vector types can be used with the `get` function, the `tuple_size` and the `tuple_element` traits and in a `structured binding <https://en.cppreference.com/w/cpp/language/structured_binding>`__ declaration.

.. rubric:: Example
.. code:: cpp
   #include <cuda/vector_types>
   #include <cuda/std/tuple>
   #include <iostream>

   template <class T>
   struct Particle
   {
     cuda::vector_type_t<T, 3> position;
     cuda::vector_type_t<T, 3> velocity;
   };

   template <class T>
   void update_particle(Particle<T>& p, const T dt)
   {
     // Graviational acceleration in y axis, creates a (0, -9.81, 0) vector
     const auto [gx, gy, gz] = cuda::make_vector<T, 3>(T{0}, T{-9.81}, T{0});

     auto& [px, py, pz] = p.position; // Decomposing the position vector
     auto& [vx, vy, vz] = p.velocity; // Decomposing the velocity vector

     // Update velocity
     vx += gx * dt;
     vy += gy * dt;
     vz += gz * dt;

     // Update position
     px += vx * dt;
     py += vy * dt;
     pz += vz * dt;
   }

   int main()
   {
     // Use single precision for the simulation
     using T = float;

     // Initial position is (0, 0, 0)
     constexpr auto initial_position = cuda::make_vector<T, 3>();

     // Initial velocity is (1, 1, 1)
     constexpr auto initial_velocity = cuda::make_vector<T, 3>(cuda::value_broadcast, T{1});

     // Simulation parameters
     constexpr auto n_steps = 100;
     constexpr auto dt = T{0.1};

     Particle<T> p{initial_position, initial_velocity};

     for (auto i = 0; i < n_steps; ++i)
     {
       update_particle(p, dt);
     }

     // Print results
     std::cout << "Initial position: (" << initial_position.x << ", " << initial_position.y << ", " << initial_position.z << ")\n";
     std::cout << "Final position: (" << p.position.x << ", " << p.position.y << ", " << p.position.z << ")\n";
   }
