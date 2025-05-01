# c.parallel type wrapper templates

This directory houses the definitions of and machinery for templates used to generate types later passed into CUB as
algorithm arguments. The entire system consists of four elements:

* argument mappings,
* the templates themselves,
* the machinery to allow referencing the above at JIT-compile time, and
* CMake machinery to turn the templates and mappings into strings usable with NVRTC.

> [!IMPORTANT]
> If you are reading this because you are about to create new jit template and/or mapping headers, **PLEASE** read at
> least the CMake section to understand when and where the `_CCCL_C_PARALLEL_JIT_TEMPLATES_PREPROCESS` macro must be
> checked for.

## Example

Here is an annotated usage of the system; please note that it requires the CMake library `cccl.c.parallel.jit_template`
to be linked with the final executable for it to work:

```cpp
#include "jit_templates/templates/input_iterator.h" // provides the input iterator template
#include "jit_templates/traits.h"                   // provides the glue machinery

// This type will be used as a tag for the specific iterator type. Having a tag like this allows multiple
// specializations of a given template to exist in the same TU without any worry about conflicts.
//
// Types used as tags have to have a name that can be used for $NAME in `struct $NAME;`, because that will be a part of
// the code NVRTC will be compiling.
struct func_iterator_tag;

void func(/*..., */ cccl_iterator_t input_it /*, ...*/) {
    // ...

    const auto [
        input_iterator_name,    // this string contains the C++ type name of the requested specialization

        input_iterator_src      // this string contains additional code that must be included in a TU using the
                                // specialization
    ] =
        get_specialization<     // get_specialization is the entry point to jit templates

            func_iterator_tag   // this is the tag type defined earlier

        >(
            // A "traits" struct provides information about the given template to `get_specialization`. It is defined
            // together with a template; for details, see the section on templates below. We pass it wrapped in
            // `template_id` to signify that it is carrying the template information.
            template_id<input_iterator_traits>(),

            // What follows are the arguments for the template. These will be transformed into simple structures
            // containing all the information from the runtime, and used as template parameters to the template specified
            // above.
            //
            // `get_specialization` performs rudimentary type checking; if you pass the wrong type of an argument here
            // (one that doesn't match the types expected by the template), this call to `get_specialization` will fail to
            // compile.
            input_t
        );

    // ...
    // When constructing the TU to be passed into NVRTC, insert the contents of the jit template headers into it, prior
    // to using any of the names obtained from `get_specialization`:
    std::string nvrtc_tu = jit_template_header_contents + rest_of_the_tu;

    // ...
}
```

## Structure of the system

### Argument mappings

An argument mapping is a type, usable as a template argument, which carries the `cccl_*` values that are the
arguments to the invocation of `get_specialization` in a format that will work in device code compiled with NVRTC.

Below is the contents of the `cccl_type_info` mapping header with annotations:

```cpp
// The mapping structure itself. This defines what is passed as template arguments.
template <typename T>
struct cccl_type_info_mapping
{
  using Type = T;
};

// For why this ifndef is necessary, please see the CMake section.
#ifndef _CCCL_C_PARALLEL_JIT_TEMPLATES_PREPROCESS
#  include "../traits.h"

// To register a mapping for a given type (generally a c.parallel type, but this can be anything), define a
// specialization of the `parameter_mapping` template declared in `traits.h`.
template <>
struct parameter_mapping<cccl_type_info>
{
  // Every parameter type must provide an archetype value. This value will be used by `get_specialization` to perform
  // its rudimentary type checking, by instantiating the given template with it as one of the template arguments.
  static const constexpr auto archetype = cccl_type_info_mapping<int>{};

  // This function defines what the mapping is. It receives the template id of the template being instantiated (this
  // allows the mapping to handle certain target templates in a special way) and the argument that was passed to
  // `get_specialization`.
  //
  // The returned string will be used in the constructed C++ type name, in the position corresponding to the position of
  // the original argument in the call to `get_specialization`.
  template <typename TplId>
  static std::string map(TplId, cccl_type_info arg)
  {
    return std::format("cccl_type_info_mapping<{}>{{}}", cccl_type_enum_to_string(arg.type));
  }

  // This function defines any additional code that is required for the value returned from `map` to be well formed when
  // compiled. It is invoked with the same arguments as `map`.
  //
  // The common use case for this feature is declaring extern functions, whose names are carried by the argument.
  template <typename TplId>
  static std::string aux(TplId, cccl_type_info)
  {
    return {};
  }
};
#endif
```

### Templates

The templates themselves define the device wrappers being produced. Their structure is entirely up to the user; however,
their template signature must follow a specific pattern.

The first template parameter of a JIT template must be a type; this will be a type with the same name as that specified
when calling `get_specialization`. It will be redeclared in device code, which means the name must be simply an
identifier.

After the tag parameter, a JIT template must accept a number of non-type template parameters equivalent to the number of
runtime arguments the author wants to be passed into `get_specialization`. Their types should correspond to the mapping
structures of the desired argument types. This way, `get_specialization` can perform basic type checking, and there can
be a 1:1 correspondence between the runtime and JIT-compile time template arguments, even if they carry more than one
piece of information (which is the typical case).

### Glue logic

There's three elements to the glue logic:

* template traits,
* parameter mapping archetypes, and
* `get_specialization`.

#### Template traits

Every JIT template must define a traits type; this is how it will be passed into `get_specialization`, and it also
allows providing custom logic for special cases.

A template traits type must provide:

1. A member template under the name `type`, which will accept parameters corresponding to the JIT template's parameters,
   and which will instantiate the JIT template with those provided arguments. Together with archetypes, this allows
   `get_specialization` to perform type checking of the runtime arguments.
2. A static member variable `name`, usable as a string. This is used as the name of the JIT template when creating the
   final type name to be returned to the user.
3. *(Optional)* A static member function called `special`, which accepts runtime arguments that must be handled
   specially, and returns an engaged `optional` containing a custom type name if any special handling occurred for the
   given arguments. The poster child for the use of this feature is the iterator JIT templates, where, if the kind of
   the iterator is determined to be a pointer, a simple pointer name is returned (instead of generating a specialization
   name of the template).

#### Archetypes

An archetype, defined within a specialization of `parameter_mapping`, is an object of (one of, if applicable) the
mapping type. This should be a reasonable stand-in for how an object of said type will look like at runtime.

The currently followed convention is that, if a runtime-known type is needed to be used, `int` is used for that type in
the archetype object (see the definition of the archetype of `cccl_type_info` quoted earlier). This will allow various
types to match if the JIT template uses those types for anything when simply instantiated.

#### `get_specialization`

Finally, there's `get_specialization`. This is the entry point of the glue logic, and the one part of the system (other
than names of JIT template traits, and the c.parallel argument types) that is end-user-facing. It receives all of the
information provided by template traits and archetypes; performs basic type checks; and returns the final type name,
plus any auxiliary code necessary to compile it. See the example at the beginning of this document for an explanation of
its various parameters and the return values.

### CMake

The last bit necessary to make all of this work is the preprocessing and embedding step. In it, the `jit_entry.h` header
file is preprocessed with the C++ compiler, with an additional definition of the macro
`_CCCL_C_PARALLEL_JIT_TEMPLATES_PREPROCESS`. The result of this preprocessing is then turned into a source file with it
as a global char array, called `jit_template_header_contents`, and compiled into a static library.

> [!IMPORTANT]
> Please read the following before authoring jit template and/or mapping headers!

The additional macro defined in this step is important for two reasons:

1. It allows to guard the host-only logic (which will often also include headers that are not available to NVRTC) from being
   visible in the code that is then passed into NVRTC. All of the template traits types and `parameter_mapping`
   specializations should be ifdef'd out if this macro is defined.
2. It allows us to control which headers are actually going to be handled by the preprocessor. We **DO NOT** want
   libcu++ headers, for instance, to end up in the final preprocessed string, as that'd lead to a lot of mess. For this
   reason, **ALL** `#include` directives, save for the ones including files in the `mappings` and `templates`
   subdirectories of this folder, must be ifdef'd out if this macro is defined (this also applies to the headers
   included by host-only logic mentioned in point 1 above).
