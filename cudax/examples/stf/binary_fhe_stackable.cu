//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief A toy example to illustrate how we can compose logical operations
 *        over encrypted data
 */

#include "cuda/experimental/__stf/utility/stackable_ctx.cuh"
#include "cuda/experimental/stf.cuh"

using namespace cuda::experimental::stf;

class ciphertext;

class plaintext {
public:
    plaintext(const stackable_ctx& ctx) : ctx(ctx) {}

    plaintext(stackable_ctx& ctx, std::vector<char> v) : values(v), ctx(ctx) {
        l = ctx.logical_data(&values[0], values.size());
    }

    void set_symbol(std::string s) {
        l.set_symbol(s);
        symbol = s;
    }

    std::string get_symbol() const { return symbol; }

    std::string symbol;

    const stackable_logical_data<slice<char>>& data() const { return l; }

    stackable_logical_data<slice<char>>& data() { return l; }

    // This will asynchronously fill string s
    void convert_to_vector(std::vector<char>& v) {
        ctx.host_launch(l.read()).set_symbol("to_vector")->*[&](auto dl) {
            v.resize(dl.size());
            for (size_t i = 0; i < dl.size(); i++) {
                v[i] = dl(i);
            }
        };
    }

    ciphertext encrypt() const;

    stackable_logical_data<slice<char>> l;

    template <typename... Pack>
    void push(Pack&&... pack) {
        l.push(::std::forward<Pack>(pack)...);
    }

    void pop() { l.pop(); }

private:
    std::vector<char> values;
    mutable stackable_ctx ctx;
};

class ciphertext {
public:
    ciphertext() = default;
    ciphertext(const ciphertext&) = default;

    ciphertext(const stackable_ctx& ctx) : ctx(ctx) {}


    plaintext decrypt() const {
        plaintext p(ctx);
        p.l = ctx.logical_data(shape_of<slice<char>>(l.shape().size()));
        // fprintf(stderr, "Decrypting...\n");
        ctx.parallel_for(l.shape(), l.read(), p.l.write()).set_symbol("decrypt")->*
                [] __device__ (size_t i, auto dctxt, auto dptxt) {
                    dptxt(i) = char((dctxt(i) >> 32));
                    // printf("DECRYPT %ld : %lx -> %x\n", i, dctxt(i), (int) dptxt(i));
                };
        return p;
    }

    // Copy assignment operator
    ciphertext& operator=(const ciphertext& other) {
        if (this != &other) {
            fprintf(stderr, "COPY ASSIGNMENT OP... this->l.depth() %ld other.l.depth() %ld - ctx depth %ld other.ctx.depth %ld\n", l.depth(), other.l.depth(), ctx.depth(), other.ctx.depth());
    //        l = ctx.logical_data(other.data().shape());
            assert(l.shape() == other.l.shape());
            other.ctx.parallel_for(l.shape(), other.l.read(), l.write()).set_symbol("copy")->*
                    [] __device__ (size_t i, auto other, auto result) { result(i) = other(i); };
        }
        return *this;
    }

    ciphertext operator|(const ciphertext& other) const {
        ciphertext result(ctx);
        result.l = ctx.logical_data(data().shape());

        ctx.parallel_for(data().shape(), data().read(), other.data().read(), result.data().write()).set_symbol("OR")->*
                [] __device__(size_t i, auto d_c1, auto d_c2, auto d_res) { d_res(i) = d_c1(i) | d_c2(i); };

        return result;
    }

    ciphertext operator&(const ciphertext& other) const {
        ciphertext result(ctx);
        result.l = ctx.logical_data(data().shape());

        ctx.parallel_for(data().shape(), data().read(), other.data().read(), result.data().write()).set_symbol("AND")->*
                [] __device__(size_t i, auto d_c1, auto d_c2, auto d_res) { d_res(i) = d_c1(i) & d_c2(i); };

        return result;
    }

    ciphertext operator~() const {
        ciphertext result(ctx);
        result.l = ctx.logical_data(data().shape());
        ctx.parallel_for(data().shape(), data().read(), result.data().write()).set_symbol("NOT")->*
                [] __device__(size_t i, auto d_c, auto d_res) { d_res(i) = ~d_c(i); };

        return result;
    }

    const stackable_logical_data<slice<uint64_t>>& data() const { return l; }

    stackable_logical_data<slice<uint64_t>>& data() { return l; }

    stackable_logical_data<slice<uint64_t>> l;

    template <typename... Pack>
    void push(Pack&&... pack) {
        l.push(::std::forward<Pack>(pack)...);
    }

    void pop() { l.pop(); }

private:
    mutable stackable_ctx ctx;
};

ciphertext plaintext::encrypt() const {
    ciphertext c(ctx);
    c.l = ctx.logical_data(shape_of<slice<uint64_t>>(l.shape().size()));

    ctx.parallel_for(l.shape(), l.read(), c.l.write()).set_symbol("encrypt")->*
            [] __device__(size_t i, auto dptxt, auto dctxt) {
                // A super safe encryption !
                dctxt(i) = ((uint64_t) (dptxt(i)) << 32 | 0x4);
            };

    return c;
}

template <typename T>
T circuit(const T& a, const T& b) {
    return (~((a | ~b) & (~a | b)));
}

int main() {
    stackable_ctx ctx;

    std::vector<char> vA { 3, 3, 2, 2, 17 };
    plaintext pA(ctx, vA);
    pA.set_symbol("A");

    std::vector<char> vB { 1, 7, 7, 7, 49 };
    plaintext pB(ctx, vB);
    pB.set_symbol("B");

    auto eA = pA.encrypt();
    auto eB = pB.encrypt();

    ctx.push_graph();

    eA.push(access_mode::read);
    eB.push(access_mode::read);

    // TODO find a way to get "out" outside of this scope to do decryption in the main ctx
    auto out = circuit(eA, eB);

    std::vector<char> v_out;
    out.decrypt().convert_to_vector(v_out);

    eA.pop();
    eB.pop();

    ctx.pop();

    ctx.finalize();

    for (size_t i = 0; i < v_out.size(); i++) {
        char expected = circuit(vA[i], vB[i]);
        EXPECT(expected == v_out[i]);
    }
}
