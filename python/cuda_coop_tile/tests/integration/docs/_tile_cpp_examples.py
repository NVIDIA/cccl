# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Numerical consumers for the C++ examples in the Tile documentation."""

from __future__ import annotations

import json

from ._tile_examples import code_blocks as blocks

HEADER = (
    "//===--------------------------------------------------------------"
    "--------===//\n"
    "//\n"
    "// Part of the CUDA Core Compute Libraries, under the Apache "
    "License v2.0 with\n"
    "// LLVM Exceptions.\n"
    "// See https://llvm.org/LICENSE.txt for license information.\n"
    "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception\n"
    "// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & "
    "AFFILIATES.\n"
    "//\n"
    "//===--------------------------------------------------------------"
    "--------===//\n"
    "#include <cuda/coop/tile.cuh>\n"
    "#include <cuda_runtime.h>\n"
    "#include <cstdlib>\n"
    "#include <iostream>\n"
    "#include <stdexcept>\n"
    "#include <vector>\n"
    "namespace ct = cuda::tiles;\n"
    "namespace coop = cuda::coop::tile;\n"
    "\n"
    "void checked(cudaError_t status) {\n"
    "  if (status != cudaSuccess) throw "
    "std::runtime_error(cudaGetErrorString(status));\n"
    "}\n"
    "template<class T> struct buffer {\n"
    "  T* data = nullptr;\n"
    "  size_t size;\n"
    "  explicit buffer(const std::vector<T>& input): size(input.size()) "
    "{\n"
    "    checked(cudaMalloc(&data, size * sizeof(T)));\n"
    "    checked(cudaMemcpy(data, input.data(), size * sizeof(T), "
    "cudaMemcpyHostToDevice));\n"
    "  }\n"
    "  ~buffer() { cudaFree(data); }\n"
    "  void print(const char* name) const {\n"
    "    std::vector<T> output(size);\n"
    "    checked(cudaMemcpy(output.data(), data, size * sizeof(T), "
    "cudaMemcpyDeviceToHost));\n"
    '    std::cout << "DATA " << name;\n'
    "    for (T value: output) std::cout << ' ' << +value;\n"
    "    std::cout << '\\n';\n"
    "  }\n"
    "};\n"
)


def marked(block):
    """Attribute compiler diagnostics to the documented source block."""

    return (
        f"#line {block['line']} {json.dumps(block['path'])}\n"
        + block["code"]
        + '\n#line 1 "documentation_consumer_scaffolding"\n'
    )


def numbers(values):
    return "{" + ",".join(str(value) for value in values) + "}"


def loads(name, size, type_name="int"):
    return (
        f"auto {name}_span = ct::tensor_span{{{name}_data, "
        f"ct::shape<{size}>{{}}}};\n"
        f"auto {name}_view = ct::partition_view{{{name}_span, "
        f"ct::shape<{size}>{{}}}};\n"
        f"auto {name} = {name}_view.load(0);\n"
    )


def consumer(block_list, inputs, outputs, *, prelude="", launch_count=1):
    """Wrap documented expressions in an executable kernel and host check.

    ``inputs`` maps each name to its C++ element type and host values.
    ``outputs`` maps each result name to its capacity. The generated program
    prints outputs so ``verify`` can check numerical results.
    Source-line directives keep diagnostics attached to the original
    documentation.
    """

    params = [
        f"const {kind}* {name}_data" for name, (kind, values) in inputs.items()
    ]
    params += [f"int* out_{name}" for name, count in outputs.items()]
    code = (
        HEADER + "\n__tile_global__ void example(" + ",".join(params) + ") {\n"
    )
    code += "\n".join(
        loads(name, len(values)) for name, (_, values) in inputs.items()
    )
    code += prelude + "\n" + "\n".join(marked(block) for block in block_list)
    for name, count in outputs.items():
        code += (
            f"auto output_span_{name} = ct::tensor_span{{out_{name}, "
            f"ct::shape<{count}>{{}}}};\n"
            f"auto output_view_{name} = "
            f"ct::partition_view{{output_span_{name}, "
            f"ct::shape<{count}>{{}}}};\n"
            f"output_view_{name}.store({name}, 0);\n"
        )
    code += "}\nint main() { try {\n"
    for name, (kind, values) in inputs.items():
        code += (
            f"buffer<{kind}> {name}(std::vector<{kind}>{numbers(values)});\n"
        )
    for name, count in outputs.items():
        code += (
            f"buffer<int> result_{name}(std::vector<int>({count}, -7654321));\n"
        )
    args = [f"{name}.data" for name in inputs] + [
        f"result_{name}.data" for name in outputs
    ]
    code += f"example<<<{launch_count},1>>>(" + ",".join(args) + ");\n"
    code += "checked(cudaGetLastError()); checked(cudaDeviceSynchronize());\n"
    for name in outputs:
        code += f'result_{name}.print("{name}");\n'
    code += (
        "return 0; } catch(const std::exception& e) { std::cerr << e.what() "
        '<< "\\n"; return 1; } }\n'
    )
    return code


def prefix(values, op, seed, inclusive=False):
    result = []
    for value in values:
        if not inclusive:
            result.append(seed)
        seed = op(seed, value)
        if inclusive:
            result.append(seed)
    return result


def primitive_case(name, block):
    data = [(i // 3) % 17 - 8 for i in range(256)]
    keys = [(i * 73) % 1021 for i in range(256)]
    values = [1000 + i for i in range(256)]
    inputs = {"input": ("int", data)}
    expected = {}
    pairs = []
    sets = []
    if name == "scan":
        data = [(-1 if i % 3 == 0 else 1) for i in range(256)]
        inputs["input"] = ("int", data)
        expected = {
            "offsets": prefix(data, lambda a, b: a + b, 0),
            "products": prefix(data, lambda a, b: a * b, 1),
            "maxima": prefix(data, max, -(2**31), True),
            "inclusive": prefix(data, lambda a, b: a + b, 0, True),
        }
    elif name == "reduce":
        expected = {
            "total": [sum(data)] * 256,
            "minimum": [min(data)] * 256,
            "shifted": [x - min(data) for x in data],
        }
    elif name == "native_reduce":
        inputs = {"values": ("int", data)}
        expected = {
            "total": [sum(data)],
            "repeated": [sum(data)] * 256,
            "shifted": [value - max(data) for value in data],
        }
    elif name == "exchange_shuffle":
        inputs["input128"] = ("int", [3 * i - 5 for i in range(128)])
        expected = {
            "blocked": [data[(i % 2) * 128 + i // 2] for i in range(256)],
            "original": data,
            "previous": [data[0]] + data[:-1],
            "rotated": [
                inputs["input128"][1][(i + 3) % 128] for i in range(128)
            ],
        }
    elif name == "adjacent_discontinuity":
        expected = {
            "left": [data[0]] + [data[i] - data[i - 1] for i in range(1, 256)],
            "right": [data[i] - data[i + 1] for i in range(255)] + [data[-1]],
            "heads": [1] + [int(data[i] != data[i - 1]) for i in range(1, 256)],
            "tails": [int(data[i] != data[i + 1]) for i in range(255)] + [1],
        }
    elif name == "histogram":
        samples = [(i * 13 + i // 7) % 79 for i in range(256)]
        inputs = {"samples": ("unsigned char", samples)}
        counts = [samples.count(i) for i in range(256)]
        expected = {"counts": counts, "same_counts": counts}
    elif name == "merge_sort":
        inputs = {"keys": ("int", keys), "values": ("int", values)}
        expected = {
            "ascending": sorted(keys),
            "ordered_keys": sorted(keys, reverse=True),
            "partial": sorted(keys[:193]) + [2147483647] * 63,
        }
        pairs = [("ordered_keys", "ordered_values", 256, 256)]
    elif name == "radix_sort_rank":
        inputs = {"keys": ("int", keys), "values": ("int", values)}
        order = sorted(range(256), key=lambda i: keys[i] & 15)
        ranks = [0] * 256
        for rank, i in enumerate(order):
            ranks[i] = rank
        expected = {
            "ordered": sorted(keys, key=lambda x: x & 255),
            "ordered_keys": sorted(keys),
            "ranks": ranks,
        }
        pairs = [("ordered_keys", "ordered_values", 256, 256)]
    elif name == "topk":
        inputs = {"keys": ("int", keys), "values": ("int", values)}
        sets = [
            ("smallest", sorted(keys)[:16]),
            ("best_keys", sorted(keys[:193], reverse=True)[:16]),
        ]
        pairs = [("best_keys", "best_values", 16, 193)]
    else:
        raise ValueError(name)
    outputs = {key: len(value) for key, value in expected.items()}
    for key, _ in sets:
        outputs[key] = 256
    for key, payload, _, _ in pairs:
        outputs[key] = 256
        outputs[payload] = 256
    return {
        "name": name,
        "blocks": [block],
        "source": consumer([block], inputs, outputs),
        "inputs": inputs,
        "expected": expected,
        "pairs": pairs,
        "sets": sets,
    }


def rld_case(block):
    values = [7, 4, 9] + [0] * 125
    lengths = [3, 1, 2] + [0] * 125
    code = HEADER + marked(block)
    code += (
        "int main() { try {\n"
        f"  buffer<int> values(std::vector<int>{numbers(values)});\n"
        "  buffer<unsigned int> lengths(std::vector<unsigned int>"
        f"{numbers(lengths)});\n"
        "  buffer<int> decoded(std::vector<int>(256, -7654321));\n"
        "  buffer<unsigned int> relative("
        "std::vector<unsigned int>(256, 12345));\n"
        "  decode_window<<<1,1>>>(values.data, lengths.data, decoded.data, "
        "relative.data, 2);\n"
        "  checked(cudaGetLastError()); checked(cudaDeviceSynchronize());\n"
        '  decoded.print("decoded"); relative.print("relative");\n'
        "  return 0;\n"
        "} catch(const std::exception& e) { std::cerr << e.what() "
        '<< "\\n"; return 1; } }\n'
    )
    return {
        "name": "run_length_decode",
        "blocks": [block],
        "source": code,
        "inputs": {
            "values": ("int", values),
            "lengths": ("unsigned int", lengths),
        },
        "expected": {
            "decoded": [7, 4, 9, 9] + [0] * 252,
            "relative": [2, 0, 0, 1] + [2**32 - 1] * 252,
        },
        "pairs": [],
        "sets": [],
    }


def guide_case(found):
    data = [-1 if i % 3 == 0 else 1 for i in range(256)]
    inputs = {"input": ("int", data)}
    expected = {
        "prefixes": prefix(data, lambda a, b: a + b, 0),
        "maximum": [max(data)] * 256,
        "raking_sum": [sum(data)] * 256,
        "blocked": [data[(i % 2) * 128 + i // 2] for i in range(256)],
        "original": data,
        "product": prefix(data, lambda a, b: a * b, 7),
    }
    return {
        "name": "programming_policies",
        "blocks": found,
        "source": consumer(
            found,
            inputs,
            {name: 256 for name in expected},
            prelude="int initial_value = 7;",
        ),
        "inputs": inputs,
        "expected": expected,
        "pairs": [],
        "sets": [],
    }


def worker_case(block):
    data = [i % 13 - 6 for i in range(256)]
    code = HEADER + marked(block) + "\nint main() { try {\n"
    code += f"buffer<int> input(std::vector<int>{numbers(data)});\n"
    code += """
  buffer<int> output128(std::vector<int>(256, -7654321));
  buffer<int> output256(std::vector<int>(256, -7654321));
  worker_sum<4><<<1,1>>>(input.data, output128.data);
  checked(cudaGetLastError()); checked(cudaDeviceSynchronize());
  worker_sum<8><<<1,1>>>(input.data, output256.data);
  checked(cudaGetLastError()); checked(cudaDeviceSynchronize());
  output128.print("sum128"); output256.print("sum256");
  return 0;
} catch(const std::exception& e) { std::cerr << e.what() << "\\n"; return 1; } }
"""
    return {
        "name": "programming_worker_warps",
        "blocks": [block],
        "source": code,
        "inputs": {"input": ("int", data)},
        "expected": {"sum128": [sum(data)] * 256, "sum256": [sum(data)] * 256},
        "pairs": [],
        "sets": [],
    }


def overview_case(block, setup):
    data = [(i % 256) % 7 + 3 * (i // 256) - 2 for i in range(512)]
    expected = [
        value
        for start in (0, 256)
        for value in prefix(data[start : start + 256], lambda a, b: a + b, 0)
    ]
    code = (
        HEADER
        + marked(setup)
        + """
__tile_global__ void overview(const int* input_data, int* output_data) {
  auto input_span = ct::tensor_span{input_data, ct::shape<512>{}};
  auto output_span = ct::tensor_span{output_data, ct::shape<512>{}};
  auto input_view = ct::partition_view{input_span, ct::shape<256>{}};
  auto output_view = ct::partition_view{output_span, ct::shape<256>{}};
"""
        + marked(block)
        + """
}
int main() { try {
"""
        + f"buffer<int> input(std::vector<int>{numbers(data)});\n"
        + """
  buffer<int> output(std::vector<int>(512, -7654321));
  overview<<<2,1>>>(input.data, output.data);
  checked(cudaGetLastError()); checked(cudaDeviceSynchronize());
  output.print("prefixes");
  return 0;
} catch(const std::exception& e) { std::cerr << e.what() << '\\n'; return 1; } }
"""
    )
    return {
        "name": "overview_two_partitions",
        "blocks": [setup, block],
        "source": code,
        "inputs": {"input": ("int", data)},
        "expected": {"prefixes": expected},
        "pairs": [],
        "sets": [],
    }


def verify(case, stdout):
    if case.get("complete_program"):
        if "PASS:" not in stdout:
            raise AssertionError("complete guide program did not report PASS")
        return
    actual = {}
    for line in stdout.splitlines():
        if line.startswith("DATA "):
            _, name, *values = line.split()
            if name in actual:
                raise AssertionError(f"duplicate output {name}")
            actual[name] = [int(value) for value in values]
    expected = dict(case["expected"])
    for name, values in expected.items():
        if actual.get(name) != values:
            raise AssertionError(
                f"{case['name']}: incorrect {name}: "
                f"{actual.get(name)} != {values}"
            )
    for name, expected_prefix in case["sets"]:
        if len(actual.get(name, [])) != 256 or sorted(
            actual[name][: len(expected_prefix)]
        ) != sorted(expected_prefix):
            raise AssertionError(
                f"{case['name']}: incorrect selected prefix {name}"
            )
    for key_name, payload_name, count, valid in case["pairs"]:
        got_keys = actual.get(key_name, [])
        payloads = actual.get(payload_name, [])
        if len(got_keys) != 256 or len(payloads) != 256:
            raise AssertionError("missing pair output")
        input_keys = case["inputs"]["keys"][1]
        input_values = case["inputs"]["values"][1]
        for key, value in zip(got_keys[:count], payloads[:count]):
            if (
                value not in input_values[:valid]
                or input_keys[input_values.index(value)] != key
            ):
                raise AssertionError(
                    f"invalid key/payload association: {key}, {value}"
                )
    checked_names = set(expected) | {name for name, _ in case["sets"]}
    checked_names |= {
        name for keys, values, _, _ in case["pairs"] for name in (keys, values)
    }
    if checked_names != actual.keys():
        raise AssertionError(
            f"unchecked or absent outputs: {checked_names ^ actual.keys()}"
        )


FINAL_NAMES = (
    "adjacent_discontinuity",
    "exchange_shuffle",
    "histogram",
    "merge_sort",
    "radix_sort_rank",
    "reduce",
    "run_length_decode",
    "scan",
    "topk",
)


def documented_cases(
    repo, docs, *, primitive_names=FINAL_NAMES, include_guide=True
):
    cases = []
    covered = set()
    exclusions = []
    assert len(set(primitive_names)) == len(primitive_names)
    assert set(primitive_names) <= set(FINAL_NAMES)
    for name in primitive_names:
        page = docs / "primitives" / f"{name}.rst"
        assert page.is_file(), page
        found = blocks(page)
        if page.stem == "run_length_decode":
            assert (
                len(found) == 2
                and "__tile_global__ void decode_window" in found[1]["code"]
            )
            cases.append(rld_case(found[1]))
            exclusions.append(
                {
                    **found[0],
                    "reason": (
                        "overload signatures with template placeholders, "
                        "not executable code"
                    ),
                }
            )
        elif page.stem == "reduce":
            native = [block for block in found if "ct::sum<0>" in block["code"]]
            repeated = [
                block for block in found if "coop::sum(input)" in block["code"]
            ]
            assert len(found) == 2 and len(native) == len(repeated) == 1, page
            cases.append(primitive_case("native_reduce", native[0]))
            cases.append(primitive_case("reduce", repeated[0]))
        else:
            assert len(found) == 1, page
            cases.append(primitive_case(page.stem, found[0]))
        covered.add(page.resolve())
    assert len(cases) == len(primitive_names) + ("reduce" in primitive_names), (
        "every selected primitive and native-reduction snippet needs a case"
    )
    guide = docs / "programming.rst"
    if include_guide:
        found = blocks(guide)
        assert len(found) == 4, "all guide C++ snippets must be covered"
        workers = [
            block
            for block in found
            if "__tile_global__ void worker_sum" in block["code"]
        ]
        assert len(workers) == 1
        cases.append(
            guide_case([block for block in found if block not in workers])
        )
        cases.append(worker_case(workers[0]))
        covered.add(guide.resolve())
        # The guide includes sections of this exact downloaded program.
        relative = "tile/cpp/test/support/tile_cpp_frontend_abi_sm120.cu"
        assert relative in guide.read_text(), "guide executable mapping changed"
        program = repo / relative
        cases.append(
            {
                "name": "programming_complete",
                "blocks": [
                    {
                        "path": str(program),
                        "line": 1,
                        "code": program.read_text(),
                    }
                ],
                "source": program.read_text(),
                "complete_program": True,
            }
        )
    index = docs / "index.rst"
    overview = blocks(index)
    installation = docs / "installation.rst"
    setup = blocks(installation)
    assert len(overview) == 1, "overview must contain its C++ partition sketch"
    assert (
        len(setup) == 1 and "#include <cuda/coop/tile.cuh>" in setup[0]["code"]
    )
    cases.append(overview_case(overview[0], setup[0]))
    covered.update((index.resolve(), installation.resolve()))
    excluded_pages = {
        (docs / "primitives" / f"{name}.rst").resolve()
        for name in set(FINAL_NAMES) - set(primitive_names)
    }
    if not include_guide:
        excluded_pages.add(guide.resolve())
    reference = docs / "api.rst"
    if reference.is_file():
        exclusions.extend(
            {
                **block,
                "reason": "API declarations, not executable kernel examples",
            }
            for block in blocks(reference)
        )
        covered.add(reference.resolve())
    for page in docs.rglob("*.rst"):
        if page.resolve() not in covered | excluded_pages and blocks(page):
            raise AssertionError(f"unmapped documentation C++ code in {page}")
    return cases, exclusions
