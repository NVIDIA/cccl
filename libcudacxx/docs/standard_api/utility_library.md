## Utility Library

Any Standard C++ header not listed below is omitted.

*: Some of the Standard C++ facilities in this header are omitted, see the
libcu++ Specifics for details.

| [`<cuda/std/type_traits>`] | Compile-time type introspection.                                                                                                             <br/><br/> 1.0.0 / CUDA 10.2 |
| [`<cuda/std/tuple>`]*      | Fixed-sized heterogeneous container (see also: [libcu++ Specifics]({{ "standard_api/utility_library/tuple.html" | relative_url }})).         <br/><br/> 1.3.0 / CUDA 11.2 |
| [`<cuda/std/functional>`]* | Function objects and function wrappers (see also: [libcu++ Specifics]({{ "standard_api/utility_library/functional.html" | relative_url }})). <br/><br/> 1.1.0 / CUDA 11.0 (Function Objects) |
| [`<cuda/std/utility>`]*    | Various utility components (see also: [libcu++ Specifics]({{ "standard_api/utility_library/utility.html" | relative_url }})).                <br/><br/> 1.3.0 / CUDA 11.2 (`pair`) |
| [`<cuda/std/version>`]     | Compile-time version information (see also: [libcu++ Specifics]({{ "standard_api/utility_library/version.html" | relative_url }})).          <br/><br/> 1.2.0 / CUDA 11.1 |
| [`<cuda/std/optional>`]*   | Optional value (see also: [libcu++ Specifics]({{ "standard_api/utility_library/optional.html" | relative_url }})).                <br/><br/> 2.3.0 / CUDA 12.4 |
| [`<cuda/std/variant>`]*    | Type safe union type (see also: [libcu++ Specifics]({{ "standard_api/utility_library/variant.html" | relative_url }})).        <br/><br/> 2.3.0 / CUDA 12.4 |
| [`<cuda/std/expected>`]*   | Optional value with error channel (see also: [libcu++ Specifics]({{ "standard_api/utility_library/expected.html" | relative_url }})).        <br/><br/> 2.3.0 / CUDA 12.4 |

[`<cuda/std/type_traits>`]: https://en.cppreference.com/w/cpp/header/type_traits
[`<cuda/std/tuple>`]: https://en.cppreference.com/w/cpp/header/tuple
[`<cuda/std/functional>`]: https://en.cppreference.com/w/cpp/header/functional
[`<cuda/std/utility>`]: https://en.cppreference.com/w/cpp/header/utility
[`<cuda/std/version>`]: https://en.cppreference.com/w/cpp/header/version
[`<cuda/std/optional>`]: https://en.cppreference.com/w/cpp/header/optional
[`<cuda/std/variant>`]: https://en.cppreference.com/w/cpp/header/variant
[`<cuda/std/expected>`]: https://en.cppreference.com/w/cpp/header/expected
