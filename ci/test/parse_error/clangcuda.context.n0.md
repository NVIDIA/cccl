📝 `value.h:90`: `error: invalid input constraint 'C' in asm`

📍 Location: `cub/cub/detail/ptx-json/value.h:90`

🎯 Target Name: cub.cpp20.detail.ptx_json.test.general.ptx

🔍 Full Error:

<pre>
  FAILED: cub/test/ptx-json/CMakeFiles/cub.cpp20.detail.ptx_json.test.general.ptx.dir/ptx_json_test_general.ptx
  /usr/bin/sccache /usr/bin/clang++  -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -D_CCCL_NO_SYSTEM_HEADER -I/home/coder/cccl/lib/cmake/cub/../../../cub -I/home/coder/cccl/lib/cmake/libcudacxx/../../../libcudacxx/include -I/home/coder/cccl/lib/cmake/thrust/../../../thrust -O3 -DNDEBUG -std=c++20 --cuda-gpu-arch=sm_80 --cuda-path=/usr/local/cuda -Xclang=-fcuda-allow-variadic-functions -Wno-unknown-cuda-version -MD -MT cub/test/ptx-json/CMakeFiles/cub.cpp20.detail.ptx_json.test.general.ptx.dir/ptx_json_test_general.ptx -MF cub/test/ptx-json/CMakeFiles/cub.cpp20.detail.ptx_json.test.general.ptx.dir/ptx_json_test_general.ptx.d -x cuda --cuda-device-only -S /home/coder/cccl/cub/test/ptx-json/ptx_json_test_general.cu -o cub/test/ptx-json/CMakeFiles/cub.cpp20.detail.ptx_json.test.general.ptx.dir/ptx_json_test_general.ptx
  In file included from /home/coder/cccl/cub/test/ptx-json/ptx_json_test_general.cu:1:
  In file included from /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/json.h:30:
  In file included from /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/array.h:30:
  In file included from /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/object.h:31:
  /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/value.h:90:29: error: invalid input constraint 'C' in asm
     90 |     asm volatile("\"%0\"" ::"C"(storage_helper<V.str[Is]...>::value) : "memory");
        |                             ^
  In file included from /home/coder/cccl/cub/test/ptx-json/ptx_json_test_general.cu:1:
  /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/json.h:50:50: error: invalid input constraint 'C' in asm
     50 |     asm volatile("cccl.ptx_json.begin(%0)\n\n" ::"C"(storage_helper<T.str[Is]...>::value) : "memory");
        |                                                  ^
  /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/json.h:52:46: error: invalid input constraint 'C' in asm
     52 |     asm volatile("\ncccl.ptx_json.end(%0)" ::"C"(storage_helper<T.str[Is]...>::value) : "memory");
        |                                              ^
  /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/json.h:50:50: error: invalid input constraint 'C' in asm
     50 |     asm volatile("cccl.ptx_json.begin(%0)\n\n" ::"C"(storage_helper<T.str[Is]...>::value) : "memory");
        |                                                  ^
  /home/coder/cccl/cub/test/ptx-json/ptx_json_test_general.cu:7:52: note: in instantiation of function template specialization 'ptx_json::tagged_json<ptx_json::string<13>{"test-json-id"}>::operator=<ptx_json::object<keyed_value<string<2>{"a"}, value<1, void>>{}, keyed_value<string<2>{"b"}, value<2, void>>{}, keyed_value<string<2>{"c"}, array<1, 2, string<2>{"a"}>>{}>, void>' requested here
      7 |   ptx_json::id<ptx_json::string("test-json-id")>() =
        |                                                    ^
  In file included from /home/coder/cccl/cub/test/ptx-json/ptx_json_test_general.cu:1:
  /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/json.h:52:46: error: invalid input constraint 'C' in asm
     52 |     asm volatile("\ncccl.ptx_json.end(%0)" ::"C"(storage_helper<T.str[Is]...>::value) : "memory");
        |                                              ^
  In file included from /home/coder/cccl/cub/test/ptx-json/ptx_json_test_general.cu:1:
  In file included from /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/json.h:30:
  In file included from /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/array.h:30:
  In file included from /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/object.h:31:
  /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/value.h:90:29: error: invalid input constraint 'C' in asm
     90 |     asm volatile("\"%0\"" ::"C"(storage_helper<V.str[Is]...>::value) : "memory");
        |                             ^
  /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/object.h:42:15: note: in instantiation of member function 'ptx_json::value<string<2>{"a"}>::emit' requested here
     42 |     value<K>::emit();
        |               ^
  /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/object.h:77:11: note: in instantiation of member function 'ptx_json::keyed_value<string<2>{"a"}, ptx_json::value<1>>::emit' requested here
     77 |     First.emit();
        |           ^
  /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/json.h:51:8: note: in instantiation of member function 'ptx_json::object<keyed_value<string<2>{"a"}, value<1, void>>{}, keyed_value<string<2>{"b"}, value<2, void>>{}, keyed_value<string<2>{"c"}, array<1, 2, string<2>{"a"}>>{}>::emit' requested here
     51 |     V::emit();
        |        ^
  /home/coder/cccl/cub/test/ptx-json/ptx_json_test_general.cu:7:52: note: in instantiation of function template specialization 'ptx_json::tagged_json<ptx_json::string<13>{"test-json-id"}>::operator=<ptx_json::object<keyed_value<string<2>{"a"}, value<1, void>>{}, keyed_value<string<2>{"b"}, value<2, void>>{}, keyed_value<string<2>{"c"}, array<1, 2, string<2>{"a"}>>{}>, void>' requested here
      7 |   ptx_json::id<ptx_json::string("test-json-id")>() =
        |                                                    ^
  In file included from /home/coder/cccl/cub/test/ptx-json/ptx_json_test_general.cu:1:
  In file included from /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/json.h:30:
  In file included from /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/array.h:30:
  In file included from /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/object.h:31:
  /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/value.h:90:29: error: invalid input constraint 'C' in asm
     90 |     asm volatile("\"%0\"" ::"C"(storage_helper<V.str[Is]...>::value) : "memory");
        |                             ^
  /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/object.h:42:15: note: in instantiation of member function 'ptx_json::value<string<2>{"c"}>::emit' requested here
     42 |     value<K>::emit();
        |               ^
  /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/object.h:78:20: note: in instantiation of member function 'ptx_json::keyed_value<string<2>{"c"}, ptx_json::array<1, 2, string<2>{"a"}>>::emit' requested here
     78 |     ((comma(), KVs.emit()), ...);
        |                    ^
  /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/json.h:51:8: note: in instantiation of member function 'ptx_json::object<keyed_value<string<2>{"a"}, value<1, void>>{}, keyed_value<string<2>{"b"}, value<2, void>>{}, keyed_value<string<2>{"c"}, array<1, 2, string<2>{"a"}>>{}>::emit' requested here
     51 |     V::emit();
        |        ^
  /home/coder/cccl/cub/test/ptx-json/ptx_json_test_general.cu:7:52: note: in instantiation of function template specialization 'ptx_json::tagged_json<ptx_json::string<13>{"test-json-id"}>::operator=<ptx_json::object<keyed_value<string<2>{"a"}, value<1, void>>{}, keyed_value<string<2>{"b"}, value<2, void>>{}, keyed_value<string<2>{"c"}, array<1, 2, string<2>{"a"}>>{}>, void>' requested here
      7 |   ptx_json::id<ptx_json::string("test-json-id")>() =
        |                                                    ^
  In file included from /home/coder/cccl/cub/test/ptx-json/ptx_json_test_general.cu:1:
  In file included from /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/json.h:30:
  In file included from /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/array.h:30:
  In file included from /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/object.h:31:
  /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/value.h:90:29: error: invalid input constraint 'C' in asm
     90 |     asm volatile("\"%0\"" ::"C"(storage_helper<V.str[Is]...>::value) : "memory");
        |                             ^
  /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/object.h:42:15: note: in instantiation of member function 'ptx_json::value<string<2>{"b"}>::emit' requested here
     42 |     value<K>::emit();
        |               ^
  /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/object.h:78:20: note: in instantiation of member function 'ptx_json::keyed_value<string<2>{"b"}, ptx_json::value<2>>::emit' requested here
     78 |     ((comma(), KVs.emit()), ...);
        |                    ^
  /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/json.h:51:8: note: in instantiation of member function 'ptx_json::object<keyed_value<string<2>{"a"}, value<1, void>>{}, keyed_value<string<2>{"b"}, value<2, void>>{}, keyed_value<string<2>{"c"}, array<1, 2, string<2>{"a"}>>{}>::emit' requested here
     51 |     V::emit();
        |        ^
  /home/coder/cccl/cub/test/ptx-json/ptx_json_test_general.cu:7:52: note: in instantiation of function template specialization 'ptx_json::tagged_json<ptx_json::string<13>{"test-json-id"}>::operator=<ptx_json::object<keyed_value<string<2>{"a"}, value<1, void>>{}, keyed_value<string<2>{"b"}, value<2, void>>{}, keyed_value<string<2>{"c"}, array<1, 2, string<2>{"a"}>>{}>, void>' requested here
      7 |   ptx_json::id<ptx_json::string("test-json-id")>() =
        |                                                    ^
  8 errors generated when compiling for sm_80.
</pre>

📝 `json.h:50`: `error: invalid input constraint 'C' in asm`

📍 Location: `cub/cub/detail/ptx-json/json.h:50`

🎯 Target Name: cub.cpp20.detail.ptx_json.test.general.ptx

🔍 Full Error:

<pre>
  FAILED: cub/test/ptx-json/CMakeFiles/cub.cpp20.detail.ptx_json.test.general.ptx.dir/ptx_json_test_general.ptx
  /usr/bin/sccache /usr/bin/clang++  -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -D_CCCL_NO_SYSTEM_HEADER -I/home/coder/cccl/lib/cmake/cub/../../../cub -I/home/coder/cccl/lib/cmake/libcudacxx/../../../libcudacxx/include -I/home/coder/cccl/lib/cmake/thrust/../../../thrust -O3 -DNDEBUG -std=c++20 --cuda-gpu-arch=sm_80 --cuda-path=/usr/local/cuda -Xclang=-fcuda-allow-variadic-functions -Wno-unknown-cuda-version -MD -MT cub/test/ptx-json/CMakeFiles/cub.cpp20.detail.ptx_json.test.general.ptx.dir/ptx_json_test_general.ptx -MF cub/test/ptx-json/CMakeFiles/cub.cpp20.detail.ptx_json.test.general.ptx.dir/ptx_json_test_general.ptx.d -x cuda --cuda-device-only -S /home/coder/cccl/cub/test/ptx-json/ptx_json_test_general.cu -o cub/test/ptx-json/CMakeFiles/cub.cpp20.detail.ptx_json.test.general.ptx.dir/ptx_json_test_general.ptx
  In file included from /home/coder/cccl/cub/test/ptx-json/ptx_json_test_general.cu:1:
  In file included from /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/json.h:30:
  In file included from /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/array.h:30:
  In file included from /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/object.h:31:
  /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/value.h:90:29: error: invalid input constraint 'C' in asm
     90 |     asm volatile("\"%0\"" ::"C"(storage_helper<V.str[Is]...>::value) : "memory");
        |                             ^
  In file included from /home/coder/cccl/cub/test/ptx-json/ptx_json_test_general.cu:1:
  /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/json.h:50:50: error: invalid input constraint 'C' in asm
     50 |     asm volatile("cccl.ptx_json.begin(%0)\n\n" ::"C"(storage_helper<T.str[Is]...>::value) : "memory");
        |                                                  ^
  /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/json.h:52:46: error: invalid input constraint 'C' in asm
     52 |     asm volatile("\ncccl.ptx_json.end(%0)" ::"C"(storage_helper<T.str[Is]...>::value) : "memory");
        |                                              ^
  /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/json.h:50:50: error: invalid input constraint 'C' in asm
     50 |     asm volatile("cccl.ptx_json.begin(%0)\n\n" ::"C"(storage_helper<T.str[Is]...>::value) : "memory");
        |                                                  ^
  /home/coder/cccl/cub/test/ptx-json/ptx_json_test_general.cu:7:52: note: in instantiation of function template specialization 'ptx_json::tagged_json<ptx_json::string<13>{"test-json-id"}>::operator=<ptx_json::object<keyed_value<string<2>{"a"}, value<1, void>>{}, keyed_value<string<2>{"b"}, value<2, void>>{}, keyed_value<string<2>{"c"}, array<1, 2, string<2>{"a"}>>{}>, void>' requested here
      7 |   ptx_json::id<ptx_json::string("test-json-id")>() =
        |                                                    ^
  In file included from /home/coder/cccl/cub/test/ptx-json/ptx_json_test_general.cu:1:
  /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/json.h:52:46: error: invalid input constraint 'C' in asm
     52 |     asm volatile("\ncccl.ptx_json.end(%0)" ::"C"(storage_helper<T.str[Is]...>::value) : "memory");
        |                                              ^
  In file included from /home/coder/cccl/cub/test/ptx-json/ptx_json_test_general.cu:1:
  In file included from /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/json.h:30:
  In file included from /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/array.h:30:
  In file included from /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/object.h:31:
  /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/value.h:90:29: error: invalid input constraint 'C' in asm
     90 |     asm volatile("\"%0\"" ::"C"(storage_helper<V.str[Is]...>::value) : "memory");
        |                             ^
  /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/object.h:42:15: note: in instantiation of member function 'ptx_json::value<string<2>{"a"}>::emit' requested here
     42 |     value<K>::emit();
        |               ^
  /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/object.h:77:11: note: in instantiation of member function 'ptx_json::keyed_value<string<2>{"a"}, ptx_json::value<1>>::emit' requested here
     77 |     First.emit();
        |           ^
  /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/json.h:51:8: note: in instantiation of member function 'ptx_json::object<keyed_value<string<2>{"a"}, value<1, void>>{}, keyed_value<string<2>{"b"}, value<2, void>>{}, keyed_value<string<2>{"c"}, array<1, 2, string<2>{"a"}>>{}>::emit' requested here
     51 |     V::emit();
        |        ^
  /home/coder/cccl/cub/test/ptx-json/ptx_json_test_general.cu:7:52: note: in instantiation of function template specialization 'ptx_json::tagged_json<ptx_json::string<13>{"test-json-id"}>::operator=<ptx_json::object<keyed_value<string<2>{"a"}, value<1, void>>{}, keyed_value<string<2>{"b"}, value<2, void>>{}, keyed_value<string<2>{"c"}, array<1, 2, string<2>{"a"}>>{}>, void>' requested here
      7 |   ptx_json::id<ptx_json::string("test-json-id")>() =
        |                                                    ^
  In file included from /home/coder/cccl/cub/test/ptx-json/ptx_json_test_general.cu:1:
  In file included from /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/json.h:30:
  In file included from /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/array.h:30:
  In file included from /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/object.h:31:
  /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/value.h:90:29: error: invalid input constraint 'C' in asm
     90 |     asm volatile("\"%0\"" ::"C"(storage_helper<V.str[Is]...>::value) : "memory");
        |                             ^
  /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/object.h:42:15: note: in instantiation of member function 'ptx_json::value<string<2>{"c"}>::emit' requested here
     42 |     value<K>::emit();
        |               ^
  /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/object.h:78:20: note: in instantiation of member function 'ptx_json::keyed_value<string<2>{"c"}, ptx_json::array<1, 2, string<2>{"a"}>>::emit' requested here
     78 |     ((comma(), KVs.emit()), ...);
        |                    ^
  /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/json.h:51:8: note: in instantiation of member function 'ptx_json::object<keyed_value<string<2>{"a"}, value<1, void>>{}, keyed_value<string<2>{"b"}, value<2, void>>{}, keyed_value<string<2>{"c"}, array<1, 2, string<2>{"a"}>>{}>::emit' requested here
     51 |     V::emit();
        |        ^
  /home/coder/cccl/cub/test/ptx-json/ptx_json_test_general.cu:7:52: note: in instantiation of function template specialization 'ptx_json::tagged_json<ptx_json::string<13>{"test-json-id"}>::operator=<ptx_json::object<keyed_value<string<2>{"a"}, value<1, void>>{}, keyed_value<string<2>{"b"}, value<2, void>>{}, keyed_value<string<2>{"c"}, array<1, 2, string<2>{"a"}>>{}>, void>' requested here
      7 |   ptx_json::id<ptx_json::string("test-json-id")>() =
        |                                                    ^
  In file included from /home/coder/cccl/cub/test/ptx-json/ptx_json_test_general.cu:1:
  In file included from /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/json.h:30:
  In file included from /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/array.h:30:
  In file included from /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/object.h:31:
  /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/value.h:90:29: error: invalid input constraint 'C' in asm
     90 |     asm volatile("\"%0\"" ::"C"(storage_helper<V.str[Is]...>::value) : "memory");
        |                             ^
  /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/object.h:42:15: note: in instantiation of member function 'ptx_json::value<string<2>{"b"}>::emit' requested here
     42 |     value<K>::emit();
        |               ^
  /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/object.h:78:20: note: in instantiation of member function 'ptx_json::keyed_value<string<2>{"b"}, ptx_json::value<2>>::emit' requested here
     78 |     ((comma(), KVs.emit()), ...);
        |                    ^
  /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/json.h:51:8: note: in instantiation of member function 'ptx_json::object<keyed_value<string<2>{"a"}, value<1, void>>{}, keyed_value<string<2>{"b"}, value<2, void>>{}, keyed_value<string<2>{"c"}, array<1, 2, string<2>{"a"}>>{}>::emit' requested here
     51 |     V::emit();
        |        ^
  /home/coder/cccl/cub/test/ptx-json/ptx_json_test_general.cu:7:52: note: in instantiation of function template specialization 'ptx_json::tagged_json<ptx_json::string<13>{"test-json-id"}>::operator=<ptx_json::object<keyed_value<string<2>{"a"}, value<1, void>>{}, keyed_value<string<2>{"b"}, value<2, void>>{}, keyed_value<string<2>{"c"}, array<1, 2, string<2>{"a"}>>{}>, void>' requested here
      7 |   ptx_json::id<ptx_json::string("test-json-id")>() =
        |                                                    ^
  8 errors generated when compiling for sm_80.
</pre>

📝 `json.h:52`: `error: invalid input constraint 'C' in asm`

📍 Location: `cub/cub/detail/ptx-json/json.h:52`

🎯 Target Name: cub.cpp20.detail.ptx_json.test.general.ptx

🔍 Full Error:

<pre>
  FAILED: cub/test/ptx-json/CMakeFiles/cub.cpp20.detail.ptx_json.test.general.ptx.dir/ptx_json_test_general.ptx
  /usr/bin/sccache /usr/bin/clang++  -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -D_CCCL_NO_SYSTEM_HEADER -I/home/coder/cccl/lib/cmake/cub/../../../cub -I/home/coder/cccl/lib/cmake/libcudacxx/../../../libcudacxx/include -I/home/coder/cccl/lib/cmake/thrust/../../../thrust -O3 -DNDEBUG -std=c++20 --cuda-gpu-arch=sm_80 --cuda-path=/usr/local/cuda -Xclang=-fcuda-allow-variadic-functions -Wno-unknown-cuda-version -MD -MT cub/test/ptx-json/CMakeFiles/cub.cpp20.detail.ptx_json.test.general.ptx.dir/ptx_json_test_general.ptx -MF cub/test/ptx-json/CMakeFiles/cub.cpp20.detail.ptx_json.test.general.ptx.dir/ptx_json_test_general.ptx.d -x cuda --cuda-device-only -S /home/coder/cccl/cub/test/ptx-json/ptx_json_test_general.cu -o cub/test/ptx-json/CMakeFiles/cub.cpp20.detail.ptx_json.test.general.ptx.dir/ptx_json_test_general.ptx
  In file included from /home/coder/cccl/cub/test/ptx-json/ptx_json_test_general.cu:1:
  In file included from /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/json.h:30:
  In file included from /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/array.h:30:
  In file included from /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/object.h:31:
  /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/value.h:90:29: error: invalid input constraint 'C' in asm
     90 |     asm volatile("\"%0\"" ::"C"(storage_helper<V.str[Is]...>::value) : "memory");
        |                             ^
  In file included from /home/coder/cccl/cub/test/ptx-json/ptx_json_test_general.cu:1:
  /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/json.h:50:50: error: invalid input constraint 'C' in asm
     50 |     asm volatile("cccl.ptx_json.begin(%0)\n\n" ::"C"(storage_helper<T.str[Is]...>::value) : "memory");
        |                                                  ^
  /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/json.h:52:46: error: invalid input constraint 'C' in asm
     52 |     asm volatile("\ncccl.ptx_json.end(%0)" ::"C"(storage_helper<T.str[Is]...>::value) : "memory");
        |                                              ^
  /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/json.h:50:50: error: invalid input constraint 'C' in asm
     50 |     asm volatile("cccl.ptx_json.begin(%0)\n\n" ::"C"(storage_helper<T.str[Is]...>::value) : "memory");
        |                                                  ^
  /home/coder/cccl/cub/test/ptx-json/ptx_json_test_general.cu:7:52: note: in instantiation of function template specialization 'ptx_json::tagged_json<ptx_json::string<13>{"test-json-id"}>::operator=<ptx_json::object<keyed_value<string<2>{"a"}, value<1, void>>{}, keyed_value<string<2>{"b"}, value<2, void>>{}, keyed_value<string<2>{"c"}, array<1, 2, string<2>{"a"}>>{}>, void>' requested here
      7 |   ptx_json::id<ptx_json::string("test-json-id")>() =
        |                                                    ^
  In file included from /home/coder/cccl/cub/test/ptx-json/ptx_json_test_general.cu:1:
  /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/json.h:52:46: error: invalid input constraint 'C' in asm
     52 |     asm volatile("\ncccl.ptx_json.end(%0)" ::"C"(storage_helper<T.str[Is]...>::value) : "memory");
        |                                              ^
  In file included from /home/coder/cccl/cub/test/ptx-json/ptx_json_test_general.cu:1:
  In file included from /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/json.h:30:
  In file included from /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/array.h:30:
  In file included from /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/object.h:31:
  /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/value.h:90:29: error: invalid input constraint 'C' in asm
     90 |     asm volatile("\"%0\"" ::"C"(storage_helper<V.str[Is]...>::value) : "memory");
        |                             ^
  /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/object.h:42:15: note: in instantiation of member function 'ptx_json::value<string<2>{"a"}>::emit' requested here
     42 |     value<K>::emit();
        |               ^
  /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/object.h:77:11: note: in instantiation of member function 'ptx_json::keyed_value<string<2>{"a"}, ptx_json::value<1>>::emit' requested here
     77 |     First.emit();
        |           ^
  /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/json.h:51:8: note: in instantiation of member function 'ptx_json::object<keyed_value<string<2>{"a"}, value<1, void>>{}, keyed_value<string<2>{"b"}, value<2, void>>{}, keyed_value<string<2>{"c"}, array<1, 2, string<2>{"a"}>>{}>::emit' requested here
     51 |     V::emit();
        |        ^
  /home/coder/cccl/cub/test/ptx-json/ptx_json_test_general.cu:7:52: note: in instantiation of function template specialization 'ptx_json::tagged_json<ptx_json::string<13>{"test-json-id"}>::operator=<ptx_json::object<keyed_value<string<2>{"a"}, value<1, void>>{}, keyed_value<string<2>{"b"}, value<2, void>>{}, keyed_value<string<2>{"c"}, array<1, 2, string<2>{"a"}>>{}>, void>' requested here
      7 |   ptx_json::id<ptx_json::string("test-json-id")>() =
        |                                                    ^
  In file included from /home/coder/cccl/cub/test/ptx-json/ptx_json_test_general.cu:1:
  In file included from /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/json.h:30:
  In file included from /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/array.h:30:
  In file included from /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/object.h:31:
  /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/value.h:90:29: error: invalid input constraint 'C' in asm
     90 |     asm volatile("\"%0\"" ::"C"(storage_helper<V.str[Is]...>::value) : "memory");
        |                             ^
  /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/object.h:42:15: note: in instantiation of member function 'ptx_json::value<string<2>{"c"}>::emit' requested here
     42 |     value<K>::emit();
        |               ^
  /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/object.h:78:20: note: in instantiation of member function 'ptx_json::keyed_value<string<2>{"c"}, ptx_json::array<1, 2, string<2>{"a"}>>::emit' requested here
     78 |     ((comma(), KVs.emit()), ...);
        |                    ^
  /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/json.h:51:8: note: in instantiation of member function 'ptx_json::object<keyed_value<string<2>{"a"}, value<1, void>>{}, keyed_value<string<2>{"b"}, value<2, void>>{}, keyed_value<string<2>{"c"}, array<1, 2, string<2>{"a"}>>{}>::emit' requested here
     51 |     V::emit();
        |        ^
  /home/coder/cccl/cub/test/ptx-json/ptx_json_test_general.cu:7:52: note: in instantiation of function template specialization 'ptx_json::tagged_json<ptx_json::string<13>{"test-json-id"}>::operator=<ptx_json::object<keyed_value<string<2>{"a"}, value<1, void>>{}, keyed_value<string<2>{"b"}, value<2, void>>{}, keyed_value<string<2>{"c"}, array<1, 2, string<2>{"a"}>>{}>, void>' requested here
      7 |   ptx_json::id<ptx_json::string("test-json-id")>() =
        |                                                    ^
  In file included from /home/coder/cccl/cub/test/ptx-json/ptx_json_test_general.cu:1:
  In file included from /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/json.h:30:
  In file included from /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/array.h:30:
  In file included from /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/object.h:31:
  /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/value.h:90:29: error: invalid input constraint 'C' in asm
     90 |     asm volatile("\"%0\"" ::"C"(storage_helper<V.str[Is]...>::value) : "memory");
        |                             ^
  /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/object.h:42:15: note: in instantiation of member function 'ptx_json::value<string<2>{"b"}>::emit' requested here
     42 |     value<K>::emit();
        |               ^
  /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/object.h:78:20: note: in instantiation of member function 'ptx_json::keyed_value<string<2>{"b"}, ptx_json::value<2>>::emit' requested here
     78 |     ((comma(), KVs.emit()), ...);
        |                    ^
  /home/coder/cccl/lib/cmake/cub/../../../cub/cub/detail/ptx-json/json.h:51:8: note: in instantiation of member function 'ptx_json::object<keyed_value<string<2>{"a"}, value<1, void>>{}, keyed_value<string<2>{"b"}, value<2, void>>{}, keyed_value<string<2>{"c"}, array<1, 2, string<2>{"a"}>>{}>::emit' requested here
     51 |     V::emit();
        |        ^
  /home/coder/cccl/cub/test/ptx-json/ptx_json_test_general.cu:7:52: note: in instantiation of function template specialization 'ptx_json::tagged_json<ptx_json::string<13>{"test-json-id"}>::operator=<ptx_json::object<keyed_value<string<2>{"a"}, value<1, void>>{}, keyed_value<string<2>{"b"}, value<2, void>>{}, keyed_value<string<2>{"c"}, array<1, 2, string<2>{"a"}>>{}>, void>' requested here
      7 |   ptx_json::id<ptx_json::string("test-json-id")>() =
        |                                                    ^
  8 errors generated when compiling for sm_80.
</pre>
