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

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

class ciphertext;

class plaintext
{
public:
  plaintext(const context& ctx)
      : ctx(ctx)
  {}

  plaintext(context& ctx, std::vector<char> v)
      : values(v)
      , ctx(ctx)
  {
    l = ctx.logical_data(&values[0], values.size());
  }

  void set_symbol(std::string s)
  {
    l.set_symbol(s);
    symbol = s;
  }

  std::string get_symbol() const
  {
    return symbol;
  }

  std::string symbol;

  const logical_data<slice<char>>& data() const
  {
    return l;
  }

  logical_data<slice<char>>& data()
  {
    return l;
  }

  // This will asynchronously fill string s
  void convert_to_vector(std::vector<char>& v)
  {
    ctx.host_launch(l.read()).set_symbol("to_vector")->*[&](auto dl) {
      v.resize(dl.size());
      for (size_t i = 0; i < dl.size(); i++)
      {
        v[i] = dl(i);
      }
    };
  }

  ciphertext encrypt() const;

  logical_data<slice<char>> l;

private:
  std::vector<char> values;
  mutable context ctx;
};

class ciphertext
{
public:
  ciphertext(const context& ctx)
      : ctx(ctx)
  {}

  plaintext decrypt() const
  {
    plaintext p(ctx);
    p.l = ctx.logical_data(shape_of<slice<char>>(l.shape().size()));
    // fprintf(stderr, "Decrypting...\n");
    ctx.parallel_for(l.shape(), l.read(), p.l.write()).set_symbol("decrypt")->*
      [] _CCCL_DEVICE(size_t i, auto dctxt, auto dptxt) {
        dptxt(i) = char((dctxt(i) >> 32));
        // printf("DECRYPT %ld : %lx -> %x\n", i, dctxt(i), (int) dptxt(i));
      };
    return p;
  }

  ciphertext operator|(const ciphertext& other) const
  {
    ciphertext result(ctx);
    result.l = ctx.logical_data(data().shape());

    ctx.parallel_for(data().shape(), data().read(), other.data().read(), result.data().write()).set_symbol("OR")->*
      [] _CCCL_DEVICE(size_t i, auto d_c1, auto d_c2, auto d_res) {
        d_res(i) = d_c1(i) | d_c2(i);
      };

    return result;
  }

  ciphertext operator&(const ciphertext& other) const
  {
    ciphertext result(ctx);
    result.l = ctx.logical_data(data().shape());

    ctx.parallel_for(data().shape(), data().read(), other.data().read(), result.data().write()).set_symbol("AND")->*
      [] _CCCL_DEVICE(size_t i, auto d_c1, auto d_c2, auto d_res) {
        d_res(i) = d_c1(i) & d_c2(i);
      };

    return result;
  }

  ciphertext operator~() const
  {
    ciphertext result(ctx);
    result.l = ctx.logical_data(data().shape());
    ctx.parallel_for(data().shape(), data().read(), result.data().write()).set_symbol("NOT")->*
      [] _CCCL_DEVICE(size_t i, auto d_c, auto d_res) {
        d_res(i) = ~d_c(i);
      };

    return result;
  }

  const logical_data<slice<uint64_t>>& data() const
  {
    return l;
  }

  logical_data<slice<uint64_t>>& data()
  {
    return l;
  }

  logical_data<slice<uint64_t>> l;

private:
  mutable context ctx;
};

ciphertext plaintext::encrypt() const
{
  ciphertext c(ctx);
  c.l = ctx.logical_data(shape_of<slice<uint64_t>>(l.shape().size()));

  ctx.parallel_for(l.shape(), l.read(), c.l.write()).set_symbol("encrypt")->*
    [] _CCCL_DEVICE(size_t i, auto dptxt, auto dctxt) {
      // A super safe encryption !
      dctxt(i) = ((uint64_t) (dptxt(i)) << 32 | 0x4);
    };

  return c;
}

template <typename T>
T circuit(const T& a, const T& b)
{
  return (~((a | ~b) & (~a | b)));
}

int main()
{
  context ctx;

  std::vector<char> vA{3, 3, 2, 2, 17};
  plaintext pA(ctx, vA);
  pA.set_symbol("A");

  std::vector<char> vB{1, 7, 7, 7, 49};
  plaintext pB(ctx, vB);
  pB.set_symbol("B");

  auto eA  = pA.encrypt();
  auto eB  = pB.encrypt();
  auto out = circuit(eA, eB);

  std::vector<char> v_out;
  out.decrypt().convert_to_vector(v_out);

  ctx.finalize();

  for (size_t i = 0; i < v_out.size(); i++)
  {
    char expected = circuit(vA[i], vB[i]);
    EXPECT(expected == v_out[i]);
  }
}
