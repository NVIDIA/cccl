# Short intro to ptx-json

This directory contains a library for embedding an almost valid JSON string in a PTX instruction stream produced with
the CUDA C++ compiler. This tool is useful for transmitting structured information from device TUs compiled with NVRTC
to the code compiling said TUs, and is used in c.parallel for figuring out the correct tuning policies for algorithms.

> [!CAUTION]
> The PTX instruction stream produced when using this library is NOT valid. If you use this library, make sure to only
> instantiate its functions when compiling to PTX and no further. You should hide their instantiations behind an opt-in
> macro for general code.

## Example code

This code is taken from a test of this library, demonstrating all the current capabilities:

```cpp
  ptx_json::id<ptx_json::string("test-json-id")>() =
    ptx_json::object<(ptx_json::key<"a">() = ptx_json::value<1>()),
                     (ptx_json::key<"b">() = ptx_json::value<2>()),
                     (ptx_json::key<"c">() = ptx_json::array<1, 2, ptx_json::string("a")>())>();
```

When compiled into PTX and extracted using a JSON parser capable of parsing C-style comments (such as `nlohmann/json`
with an opt-in argument), this produces a JSON equivalent to

```json
{
    "a": 1,
    "b": 2,
    "c": [1, 2, "a"]
}
```

Let's take the elements of this snippet apart.

### `ptx_json::string`

For a number of reasons related to how NVCC treats string arguments to templates, strings passed to other constructs of
the library must be passed around using `ptx_json::string` as seen above. The one exception to this rule is
`ptx_json::key`, which transparently creates a `ptx_json::string` object when passed a string literal.

### `ptx_json::value`

This type is used to lift core constant expressions into the type system. All constants must be passed around using
`ptx_json::value` as seen above. The one exception to this rule is `ptx_json::array`, which transparently turns its
template arguments into `ptx_json::value` objects.

Currently supported types are:
* `int`
* `ptx_json::string`

### `ptx_json::array`

This is an abstraction over a JSON array. All the template arguments to this type will be emitted as a comma-separated
list of elements between square brackets in the resulting JSON block. The arguments may be just constants or other types
provided by this library, other than `ptx_json::key` and `ptx_json::id`. This includes `ptx_json::array` and
`ptx_json::object`, for deeper structures.

### `ptx_json::object` (and `ptx_json::key`)

This is an abstraction over a JSON object. It accepts a list of keyed elements as its template arguments; the special
`key` type is used to associate a string key with a value to be emitted under that key. The associated value must be an
object provided by this library other than `ptx_json::key` or `ptx_json::id`. This includes `ptx_json::array` and
`ptx_json::object`, for deeper structures.

### `ptx_json::id`

This is the second new construct added on top of JSON for the purposes of this library. It emits whatever JSON value is
assigned to it between two tags recognized by the included parser (see `ptx-json-parser.h`) as the beginning and the
ending of a JSON block, additionally marked with a user provided ID, allowing for multiple JSON blocks within the same
TU to be distinguished.
