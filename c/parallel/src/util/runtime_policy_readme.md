# Runtime policy extraction from CUB

> [!NOTE]
> This document assumes familiarity with the `cub/detail/ptx-json` library. Please see [the `ptx-json`
> readme](/cub/cub/detail/ptx-json/README.md) for more information.

To avoid duplicating the definitions of runtime policies between CUB and c.parallel, we use the `ptx-json` library to
transmit well-structured policy information from device TUs to the c.parallel build functions. This document describes
the elements of this system, and its impact on both c.parallel and CUB.

For the purpose of this document, the following phrases are defined:
* "CUB-style policy" refers to a compile-time only (algorithm or agent) policy, which uses static member variables
  (written in `CONSTANT_CASE`) for values, and static member typedefs for nested information (written in `CamelCase`).
* "Function-based policy" refers to a compile- or run-time (algorithm or agent) policy, which uses member functions for
  accessing both values and nested information (all written in `CamelCase`).

## Overview of the system

1. CUB defines agent policy categories for different algorithms. This step uses the `CUB_DETAIL_POLICY_WRAPPER_DEFINE`
   macro.

2. CUB defines algorithm policy wrappers.

3. c.parallel build functions construct a TU containing all code necessary for a policy wrapper instantiation to
   compile, and constructs an expression naming an object of an algorithm policy wrapper type.

4. c.parallel build functions use the `get_policy` function, declared in
   [`runtime_policy.h`](/c/parallel/src/util/runtime_policy.h), to compile its definition to PTX and obtain a
   `nlohmann::json` object containing the information in the relevant policy in the device TU.

5. c.parallel build functions extract the relevant subpolicies (using the `from_json` functions described later) into
   policy objects and strings containing the definitions of equivalent policies for device code.

6. c.parallel build functions use the aforementioned strings in the construction of a final TU, which allows it to
   compile the correct versions of device kernels.

7. c.parallel build functions save the runtime policy objects in the build result.

8. c.parallel launch functions pass the saved runtime policies into the CUB dispatch layers, together with kernels
   compiled using the same policies.

## CUB policy wrapper types

### Algorithm policy wrappers

An algorithm policy generally contains one or more agent sub-policies used in different cases and kernels by said
algorithm. In CUB, these use member typedefs to define specific agent policies; to enable passing runtime values into
CUB algorithm dispatch layers, algorithm-specific policy wrappers have to be defined. An example of such an algorithm
policy wrapper is the one defined for `DeviceReduce`.

First, we define a primary template that simply inherits from the provided policy:

```cpp
template <typename PolicyT, typename = void>
struct ReducePolicyWrapper : PolicyT
{
  CUB_RUNTIME_FUNCTION ReducePolicyWrapper(PolicyT base)
      : PolicyT(base)
  {}
};
```

This definition will be used with any policy that does not match a CUB-style one, including runtime policies provided
from elsewhere. It is the responsibility of the authors of those to define all the necessary member functions correctly.

Next, we define a specialization that matches a CUB-style reduce policy. These always have member typedefs
`ReducePolicy`, `SingleTilePolicy`, and `SegmentedReducePolicy`:

```cpp
template <typename StaticPolicyT>
struct ReducePolicyWrapper<StaticPolicyT,
                           ::cuda::std::void_t<typename StaticPolicyT::ReducePolicy,
                                               typename StaticPolicyT::SingleTilePolicy,
                                               typename StaticPolicyT::SegmentedReducePolicy>> : StaticPolicyT
{
  CUB_RUNTIME_FUNCTION ReducePolicyWrapper(StaticPolicyT base)
      : StaticPolicyT(base)
  {}
```

Also in this specialization, we use the `CUB_DEFINE_SUB_POLICY_GETTER` macro to easily define functions that return the
aforementioned subpolicies using the function style:

```cpp
  CUB_DEFINE_SUB_POLICY_GETTER(Reduce)
  CUB_DEFINE_SUB_POLICY_GETTER(SingleTile)
  CUB_DEFINE_SUB_POLICY_GETTER(SegmentedReduce)
```

The `CUB_DEFINE_SUB_POLICY_GETTER` macro also automates wrapping the subpolicies in agent policy wrappers; see the next
section for more details.

Finally, because the functions defined above return wrapped policies, we also have access to their `EncodedPolicy`
functions. These return `ptx-json` objects, which we can combine into a larger object describing all the subpolicies of
the algorithm:

```cpp
#if defined(CUB_ENABLE_POLICY_PTX_JSON)
  _CCCL_DEVICE static constexpr auto EncodedPolicy()
  {
    using namespace ptx_json;
    return object<key<"ReducePolicy">()          = Reduce().EncodedPolicy(),
                  key<"SingleTilePolicy">()      = SingleTile().EncodedPolicy(),
                  key<"SegmentedReducePolicy">() = SegmentedReduce().EncodedPolicy()>();
  }
#endif
};
```

### Agent policy wrappers

CUB provides a macro, `CUB_DETAIL_POLICY_WRAPPER_DEFINE`, defined in `cub/util_device.cuh`, which defines several types
and functions all related to obtaining the policy information and using it at runtime. Which specific types are defined
depends on what configuration macros are defined when compiling CUB; these macros are mentioned where a type or function
is described if they are necessary to enable its generation.

The macro takes the following form:

```cpp
CUB_DETAIL_POLICY_WRAPPER_DEFINE(
  GenericAgentPolicy, // policy category name
  (always_true), // policy categories of which this category is a specialization of; always_true is special

  // tuples defining fields of a given policy category:
  (BLOCK_THREADS, BlockThreads, int),
  (ITEMS_PER_THREAD, ItemsPerThread, int)
)
```

This invocation defines the `GenericAgentPolicy` agent policy category. This is the name for which `<Name>` in the
following sections stands for. Specifying `always_true` as the "parent" category means we are defining a root category;
when defining a new category, make sure to not create ambiguities between unrelated categories, or compilation errors
will ensue.

After the name and the list of parent categories follows a list of tuples describing the fields of a given policy
category. Each tuple must contain three elements:
* the CUB-style name of the field;
* the name of a corresponding function in the function-based style; and
* the type of the field.

> [!TIP]
> Although not a part of the interface of this macro, in C++20 and above, a concept using this same name is also
> defined. This concept is used to constrain the overloads of `MakePolicyWrapper`, and the list of "parent" categories
> is used in the definition of the concept to create subsumption relationships to avoid ambiguities between those
> functions. This is the mental model you should follow when using this macro.

The following entities are defined when the above macro invocation is expanded:
* `GenericAgentPolicy`, a concept; do not rely on this outside of the usage of its name in the list of "parent"
  categories;
* `GenericAgentPolicyWrapper`, a wrapper translating a CUB-style policy to a function-based one; and
* `RuntimeGenericAgentPolicy`, a runtime function-based policy type; this is only defined if `CUB_DEFINE_RUNTIME_POLICIES`
  is defined.

> [!NOTE]
> For the purpose of code examples in the following sections, the name of the wrapped CUB policy is assumed to be
> `StaticPolicyT`; this is also the name for it used in the definition of the macro.

> [!TIP]
> For real invocations of this macro in code, see [`util_device.cuh`](/cub/cub/util_device.cuh) and
> [`agent_reduce.cuh`](/cub/cub/agent/agent_reduce.cuh).

### `<Name>Wrapper`

This type is always defined when invoking the macro.

This type is used to adapt a CUB-style static member style of defining values of the policy fields to the member
function style needed for c.parallel to be able to provide runtime policies into CUB's dispatch functions.

#### Field accessors

For each policy field provided to the macro, this wrapper gets a function, named according to the second element of the
field description tuple, which returns that field's value. For instance, for a field defined as `(CONSTANT_VALUE,
ConstantValue, int)`, the `<Name>Wrapper` struct is going to have the following static member function:

```cpp
__host__ __device__ static constexpr auto ConstantValue() {
    return StaticPolicyT::CONSTANT_VALUE;
}
```

#### `MakePolicyWrapper` (free function)

This function is always defined when invoking the macro.

This function is defined for each policy category, and allows for easy turning of CUB-style policies into the
function-based policies from generic code.

> [!WARNING]
> Because of the use of concept subsumption for the purpose of making the various `MakePolicyWrapper` overloads
> unambiguous, invocations of `CUB_DETAIL_POLICY_WRAPPER_DEFINE` that define more specialized categories of policies
> should be guarded with `#if defined(CUB_DEFINE_RUNTIME_POLICIES) || defined(CUB_ENABLE_POLICY_PTX_JSON)`, until C++20
> is enabled CCCL-wide.

#### `EncodedPolicy` (static member function)

This function is only defined if `CUB_ENABLE_POLICY_PTX_JSON` is defined.

This function returns a `ptx_json::object` object that encodes the agent policy described by the `StaticPolicyT` object.
Currently it also contains a `__dummy` key, but this will be removed once C++20 is enabled CCCL-wide.

### `Runtime<Name>`

This type is only defined if `CUB_DEFINE_RUNTIME_POLICIES` is defined.

This type is used at runtime to provide CUB with runtime-determined policies. It also allows turning a `nlohmann::json`
object into a runtime instance of the policy in question.

#### Fields and field accessors

Similarly to the `<Name>Wrapper` types, the `Runtime<Name>` types provide function style accessors to the policy fields;
however, in this case, they are non-static and non-constexpr member functions, and return the values of members of the
runtime wrapper.

For instance, for a field defined as `(CONSTANT_VALUE, ConstantValue, int)`, the `Runtime<Name>` struct is going to
have the following members:

```cpp
int runtime_CONSTANT_VALUE;
int ConstantValue() const {
    return runtime_CONSTANT_VALUE;
}
```

#### `from_json` (static member function)

This function accepts an `nlohmann::json` object describing an algorithm policy, which under a subpolicy name also
provided to this function contains a agent policy matching the category of the defined wrappers, and returns an instance
of `Runtime<Name>` filled in with those values. It *also* returns a string that can be used in an NVRTC translation unit
to define a CUB-style policy with those same values.
