.. _systems:

Thrust systems
==============

Thrust offers a set of algorithms and APIs which can dispatch to various systems.
A system is basically a backend and Thrust currently supports the following systems:

 - cpp
 - cuda
 - omp
 - tbb
 - generic
 - sequential

The generic and sequential systems are implementation details.
Users can define additional systems to add new backends.

Each system lives in a directory under ``thrust/system/[detail/]``.


Execution policy base classes
*****************************

Thrust defines common base classes for execution policies:

.. code-block:: c++

    namespace thrust::detail {
      struct execution_policy_marker {};

      template <typename DerivedPolicy>
      struct execution_policy_base : execution_policy_marker {};
    }
    namespace thrust {
      template <typename DerivedPolicy>
      struct execution_policy : thrust::detail::execution_policy_base<DerivedPolicy> {};
    }

There is an execution policy marker, which sits at the top of the inheritance chain.
Then, we have an execution policy base and the actual execution policy,
both are templated on the derived policy type (CRTP).


System execution policy base classes and tags
*********************************************

Inside each system directory is a header file ``execution_policy.h``
which defines the execution policy and tag for that system,
except for the generic system, which does not have a dedicated execution policy.
The inheritance for a system, e.g. ``cpp``, looks like this:

.. code-block:: c++

    namespace thrust::system::cpp {
      struct tag; // forward declaration

      template <typename Derived>
      struct execution_policy; // forward declaration

      template <>
      struct execution_policy<tag> : ... {};

      struct tag : execution_policy<tag> {};

      template <typename Derived>
      struct execution_policy : ... {
        using tag_type = tag;
        _CCCL_HOST_DEVICE operator tag() const { return {}; }
      };
    }

Each system has it's own execution policy, again templated on a further derived execution policy.
The system's execution policy derives (directly or indirectly) from ``thrust::execution_policy``.
System execution policies are templates and intended to be further derived from.
Additionally, there is a tag for each system, without any template parameters,
that derives from the system's execution policy.
Tags are non-template class types.
The system's execution policy is specialized for the tag type to have no members,
otherwise it has an alias for the tag type and can convert to the tag type.
Therefore, the execution policy can always be converted to a tag (either by downcasting or by a conversion).

Various systems now further extend this hierarchy of execution policies, or play other tricks.
The ``cpp::execution_policy<Derived>`` inherits from ``sequential::execution_policy<Derived>`` for example
(which then inherits from ``thrust::execution_policy``),
so any dispatch to an algorithm in the cpp system may fall back to the sequential system.
The cuda tag additionally inherits from ``allocator_aware_execution_policy`` to provide further functionality.


Parallel and sequential policy
******************************

Each system also defines an internal parallel policy ``thrust::system::*::detail::par_t``.
The cpp system for example:

.. code-block:: c++

    namespace thrust::system::cpp {
      namespace detail {
        struct par_t : execution_policy<par_t>, ... {};
      }
      inline constexpr detail::par_t par;
    }

These policies can be used by a user directly, to pick an execution order with a specific backend.
Some systems also provide additional parallel execution policies,
or member functions which can further configure a policy.
The CUDA system for example also provides ``par_nosync_t``
or can customize ``par_t`` by calling ``par.on(stream)``.
In any case, the type passed to a Thrust algorithm will always be
a class derived from the system's ``execution_policy`` class template.

Thrust further defines a single sequential policy, ``thrust::seq``:

.. code-block:: c++

    namespace thrust {
      namespace detail {
        struct seq_t : system::detail::sequential::execution_policy<seq_t>, ... { ... };
      }
      inline constexpr detail::seq_t seq;
    }

which is a global constant of the execution policy to the sequential system.


Host and device system policy
*****************************

Thrust additionally defines an active host and device system,
which are selected by the macros ``THRUST_HOST_SYSTEM`` and ``THRUST_DEVICE_SYSTEM``
and alias to the corresponding system parallel policies:

.. code-block:: c++

    namespace thrust {
      namespace detail {
        using host_t   = thrust::__THRUST_HOST_SYSTEM_NAMESPACE::detail::par_t;
        using device_t = thrust::__THRUST_DEVICE_SYSTEM_NAMESPACE::detail::par_t;
      }
      inline constexpr detail::host_t host;
      inline constexpr detail::device_t device;
    }

Users most often use ``thrust::host`` and ``thrust::device`` to dispatch to the current host or device system.


Algorithm dispatch
******************

Each Thrust algorithm overload requires an execution policy to determine the backend to use.
The policy can either be specified as a first argument by the user,
or determined from the other arguments.
We will focus on the first case for now.
Let's take the public API entry point ``thrust::sort`` as an example.

.. code-block:: c++

    namespace thrust {
      template <typename DerivedPolicy, typename RandomAccessIterator>
      void sort(const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
                RandomAccessIterator first, RandomAccessIterator last);
    }

We can see that the first argument is a reference to ``execution_policy_base``,
the highest base class in the execution policy hierarchy
that still carries compile-time information on the most derived type.
This ensures that this overload is only selected, when the user passes a valid execution policy.
For comparison, C++17 parallel algorithms use a plain template parameter for the execution policy
and apply a constraint (SFINAE or requires).
The reference is also ``const`` so users can pass a temporary execution policy object,
which was just created at the call site.

Let's have a look at the implementation of the public API entry point:

.. code-block:: c++

    namespace thrust {
      template <typename DerivedPolicy, typename RandomAccessIterator>
      void sort(const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
                RandomAccessIterator first, RandomAccessIterator last) {
        using thrust::system::detail::generic::sort;
        return sort(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last);
      }
    }

We first bring the generic sort implementation from the generic system into scope.
Then, we strip away ``const`` and cast the reference to the execution policy to the most derived type,
and perform an unqualified call to ``sort`` with the same arguments apart from the execution policy.

We have previously seen that execution policies form deeper inheritance chains,
and some systems inherit the policy of other systems (e.g. the cpp system inherits the sequential system).
The ``derived_cast`` makes sure we perform ADL (argument dependent lookup) using the most specialized execution policy
when we try to find overloads of ``sort``.
It's also necessary, because when an execution policy is passed to the public API,
it binds to the reference of its base class ``execution_policy_base``,
for which no backend system exists,
so we have to bring the type down again the inheritance chain.

ADL will find a set of overloads for ``sort`` depending on the type of the execution policy.
This set will at least include ``sort`` from the generic system and the API entry point itself.
In case of the cpp system, it will also find ``sort`` from the sequential and cpp system.
The compiler then ranks the overloads and the best match is the overload from the most specialized execution policy.

This is neat, because a system does not need to provide implementations of all algorithms.
It can just fall back to a generic implementation (falling back to a different algorithm),
or to an implementation from a different system.
For example, ``thrust::count`` is not implemented in the cpp system,
so it falls back to the generic implementation, which uses ``thrust::count_if``.
That's also not implemented in the cpp system, so it falls back again to the generic system,
which then implements it via ``thrust::transform_reduce``, and so on.
As a different example, ``thrust::copy`` for the cpp system brings in the include of the sequential copy implementation,
so ADL will find it and prefer it over the generic implementation.

Any generic algorithm is always outranked by a system specific implementation
due to the inheritance chain of execution policies.
Let's look at the generic sort implementation's interface:

.. code-block:: c++

    namespace thrust::system::detail::generic {
      template <typename DerivedPolicy, typename RandomAccessIterator>
      void sort(thrust::execution_policy<DerivedPolicy>& exec,
                RandomAccessIterator first, RandomAccessIterator last);
    }

Notice that it takes the execution policy argument as ``thrust::execution_policy``,
which is derived from ``thrust::detail::execution_policy_base``, which appears in the public API.
This is why any overload in the generic system will always outrank the public API entry point.


System selection
****************

Thrust also provides overloads of most algorithms without an execution policy,
in which case the execution policy is determined based on the remaining arguments, usually iterators.
Let's look at ``thrust::adjacent_difference``:

.. code-block:: c++

    namespace thrust {
      template <typename InputIterator, typename OutputIterator>
      OutputIterator adjacent_difference(InputIterator first, InputIterator last, OutputIterator result) {
        using system::detail::generic::select_system;
        using System1 = iterator_system_t<InputIterator>;
        using System2 = iterator_system_t<OutputIterator>;
        System1 system1;
        System2 system2;
        return thrust::adjacent_difference(select_system(system1, system2), first, last, result);
      }
    }

Such an API is implemented by first bringing ``select_system`` from the generic system into scope.
Then, we determine the system types associated with all iterator types via ``thrust::iterator_system``,
and instantiate these systems.
We then select one of these systems and pass the it to the corresponding overload of ``adjacent_difference``,
taking an execution policy as first argument.
Notice that this call is qualified with ``thrust::``, so ADL is not used here.
The dispatch to the correct system will be performed in the called overload of ``adjacent_difference``.

``select_system`` is implemented in the generic system, but no other Thrust system provides a different version of it.
However, since users can define their own systems,
they could also provide a different algorithm for selecting between multiple systems.
The generic implementation will select the system to which all other systems are convertible
(this is called the minimum system).
If we remember how execution policies and tags are defined,
they form inheritance hierarchies and tags have conversion operators,
so those play a role here.
``select_system`` may not find a minimum system,
in which case it returns ``thrust::detail::unrelated_systems<System1, System2, ...>``,
which usually fails to find an overload via ADL and lead to a compilation error.
