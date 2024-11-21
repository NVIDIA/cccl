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
 * @brief This shows how we can asynchronously compose a sequence of operations
 *        described with an IR
 */

#include <cuda/experimental/stf.cuh>

#include <fstream>
#include <map>

using namespace cuda::experimental::stf;

using logical_slice = logical_data<slice<double>>;

static __global__ void cuda_sleep_kernel(long long int clock_cnt)
{
  long long int start_clock  = clock64();
  long long int clock_offset = 0;
  while (clock_offset < clock_cnt)
  {
    clock_offset = clock64() - start_clock;
  }
}

void cuda_sleep(double ms, cudaStream_t stream)
{
  int device;
  cudaGetDevice(&device);

  // cudaDevAttrClockRate: Peak clock frequency in kilohertz;
  int clock_rate;
  cudaDeviceGetAttribute(&clock_rate, cudaDevAttrClockRate, device);

  long long int clock_cnt = (long long int) (ms * clock_rate);
  cuda_sleep_kernel<<<1, 1, 0, stream>>>(clock_cnt);
}

const double sleep_time = 1.0;

// z = LITERAL(length, value)
template <typename Ctx>
logical_slice LITERAL(Ctx& ctx, size_t n, int, std::string out_symbol = "undefined")
{
  auto z = ctx.logical_data(shape_of<slice<double>>(n));
  z.set_symbol(out_symbol);

  ctx.task(z.write()).set_symbol("LITERAL")->*[](cudaStream_t stream, auto /*unused*/) {
    cuda_sleep(sleep_time, stream);
  };

  return z;
}

// z = OR(x,y)
template <typename Ctx>
logical_slice OR(Ctx& ctx, logical_slice x, logical_slice y, std::string out_symbol = "undefined")
{
  assert(x.shape().size() == y.shape().size());

  auto z = ctx.logical_data(x.shape());
  z.set_symbol(out_symbol);

  ctx.task(x.read(), y.read(), z.write()).set_symbol("OR")->*
    [](cudaStream_t stream, auto /*unused*/, auto /*unused*/, auto /*unused*/) {
      cuda_sleep(sleep_time, stream);
    };

  return z;
}

// z = AND(x,y)
template <typename Ctx>
logical_slice AND(Ctx& ctx, logical_slice x, logical_slice y, std::string out_symbol = "undefined")
{
  assert(x.shape().size() == y.shape().size());

  auto z = ctx.logical_data(x.shape());
  z.set_symbol(out_symbol);

  ctx.task(x.read(), y.read(), z.write()).set_symbol("AND")->*
    [](cudaStream_t stream, auto /*unused*/, auto /*unused*/, auto /*unused*/) {
      cuda_sleep(sleep_time, stream);
    };

  return z;
}

template <typename Ctx>
logical_slice
ARRAY_INDEX(Ctx& ctx, logical_slice x, logical_slice /*unused*/, size_t sz, std::string out_symbol = "undefined")
{
  auto z = ctx.logical_data(shape_of<slice<double>>(sz));
  z.set_symbol(out_symbol);

  ctx.task(x.read(), z.write()).set_symbol("ARRAY INDEX")->*[](cudaStream_t stream, auto /*unused*/, auto /*unused*/) {
    cuda_sleep(sleep_time, stream);
  };

  return z;
}

// z = BIT_SLICE(x, position, size)
template <typename Ctx>
logical_slice BIT_SLICE(Ctx& ctx, logical_slice x, size_t /*unused*/, size_t sz, std::string out_symbol = "undefined")
{
  auto z = ctx.logical_data(shape_of<slice<double>>(sz));
  z.set_symbol(out_symbol);

  ctx.task(x.read(), z.write()).set_symbol("BIT SLICE")->*[](cudaStream_t stream, auto /*unused*/, auto /*unused*/) {
    cuda_sleep(sleep_time, stream);
  };

  return z;
}

// y = NOT(x)
template <typename Ctx>
logical_slice NOT(Ctx& ctx, logical_slice x, std::string out_symbol = "undefined")
{
  auto y = ctx.logical_data(x.shape());
  y.set_symbol(out_symbol);

  ctx.task(x.read(), y.write()).set_symbol("NOT")->*[](cudaStream_t stream, auto /*unused*/, auto /*unused*/) {
    cuda_sleep(sleep_time, stream);
  };

  return y;
}

// y = CONCAT(sz, vector<> inputs)
template <typename Ctx>
logical_slice CONCAT(Ctx& ctx, size_t sz, std::vector<logical_slice> inputs, std::string out_symbol = "undefined")
{
  auto y = ctx.logical_data(shape_of<slice<double>>(sz));
  y.set_symbol(out_symbol);

  auto t = ctx.task();
  t.add_deps(y.write());
  t.set_symbol("CONCAT");
  for (auto& input : inputs)
  {
    t.add_deps(input.read());
  }

  t->*[](cudaStream_t stream) {
    cuda_sleep(sleep_time, stream);
  };

  return y;
}

template <typename Ctx>
void run(const char* inputfile)
{
  // Find the handle from its symbol
  std::map<std::string, logical_slice> logical_slices;
  std::string output_data_symbol;
  std::ifstream read(inputfile);
  Ctx ctx;

  // Indicates if we are parsing the body of the circuit
  bool in_body = false;

  for (std::string line; std::getline(read, line);)
  {
    std::stringstream ss(line);

    //        std::cout << "LINE : " << line << std::endl;
    if (!in_body)
    {
      std::string token;
      ss >> token;
      //            std::cout << "TOKEN : " << token << std::endl;

      if (token == "fn")
      {
        // We are parsing the declaration of the function, this starts the body
        // std::cout << "GOT DECLARATION " << line << std::endl;
        in_body = true;

        // Look for parameters
        size_t begin_params, end_params;
        begin_params       = line.find('(');
        end_params         = line.find(')');
        std::string params = line.substr(begin_params + 1, end_params - begin_params - 1);

        // std::cout << "PARAMS = " << params << std::endl;

        // Parse parameters which are separated by a comma, format = "symbol: type"
        while (true)
        {
          // Find symbol
          size_t pos;
          pos                = params.find(":");
          std::string symbol = params.substr(0, pos);
          // std::cout << symbol << std::endl;

          // We create a dummy allocation so that the data handles refers to actually allocated host memory
          double* dummy     = new double[1];
          auto param_handle = ctx.logical_data(make_slice(dummy, 1));

          param_handle.set_symbol(symbol);
          logical_slices[symbol] = param_handle;

          pos = params.find(", ");
          if (pos == std::string::npos)
          {
            break;
          }

          params.erase(0, pos + 2);
        }
      }
    }
    else
    {
      std::string token;
      ss >> token;
      // std::cout << "TOKEN : " << token << std::endl;

      if (token == "}")
      {
        // This closes the body
        in_body = false;
        continue;
      }

      // We expect lines of the format : "  symbol: type = gate_name(..., id=VALUE)"
      size_t end_symbol = line.find(":");

      // We look for the first "= " to find the gate name
      size_t gate_symbol_pos = line.find("= ");
      std::string gate       = line.substr(gate_symbol_pos + 2);

      size_t gate_name_end           = gate.find("(");
      std::string gate_symbol        = gate.substr(0, gate_name_end);
      std::string gate_args          = gate.substr(gate_name_end + 1, gate.size() - gate_name_end - 2);
      std::string gate_outvar_symbol = line.substr(2, end_symbol - 2);

      // Possibly remove the "ret" out of the gate_outvar_symbol
      size_t ret_pos = gate_outvar_symbol.find("ret ");
      if (ret_pos != std::string::npos)
      {
        // This is our result !
        gate_outvar_symbol.erase(4);

        output_data_symbol = gate_outvar_symbol;
      }

      // std::cout << "GATE OUT SYMBOL " << gate_outvar_symbol << std::endl;
      // std::cout << "GATE DESCRIPTION : " << gate << std::endl;
      // std::cout << "GATE SYMBOL " << gate_symbol << std::endl;
      // std::cout << "GATE ARGS " << gate_args << std::endl;

      // We now dispatch between the different gates
      if (gate_symbol == "literal")
      {
        //  literal.916: bits[1] = literal(value=1, id=916)
        int value = 42; // TODO parse
        size_t sz = 1;

        logical_slices[gate_outvar_symbol] = LITERAL(ctx, sz, value, gate_outvar_symbol);

        continue;
      }

      if (gate_symbol == "or")
      {
        //  or.1268: bits[1] = or(or.1267, and.1236, id=1268)
        size_t pos;
        pos                     = gate_args.find(", ");
        std::string symbol_left = gate_args.substr(0, pos);
        gate_args.erase(0, pos + 2);

        pos                      = gate_args.find(", ");
        std::string symbol_right = gate_args.substr(0, pos);
        gate_args.erase(0, pos + 2);

        // std::cout << "OR GATE on symbols" << symbol_left << " AND " << symbol_right << std::endl;

        auto data_left                     = logical_slices[symbol_left];
        auto data_right                    = logical_slices[symbol_right];
        logical_slices[gate_outvar_symbol] = OR(ctx, data_left, data_right, gate_outvar_symbol);

        continue;
      }

      if (gate_symbol == "and")
      {
        size_t pos;
        pos                     = gate_args.find(", ");
        std::string symbol_left = gate_args.substr(0, pos);
        gate_args.erase(0, pos + 2);

        pos                      = gate_args.find(", ");
        std::string symbol_right = gate_args.substr(0, pos);
        gate_args.erase(0, pos + 2);

        auto data_left                     = logical_slices[symbol_left];
        auto data_right                    = logical_slices[symbol_right];
        logical_slices[gate_outvar_symbol] = AND(ctx, data_left, data_right, gate_outvar_symbol);

        continue;
      }

      if (gate_symbol == "bit_slice")
      {
        // bit_slice.936: bits[1] = bit_slice(y, start=15, width=1, id=936)
        size_t pos;
        pos                   = gate_args.find(", ");
        std::string symbol_in = gate_args.substr(0, pos);
        gate_args.erase(0, pos + 2);

        // harcoded ...
        size_t sz = 1;

        auto data_in                       = logical_slices[symbol_in];
        logical_slices[gate_outvar_symbol] = BIT_SLICE(ctx, data_in, 42, sz, gate_outvar_symbol);

        // std::cout << "PRODUCED DATA FOR " << gate_outvar_symbol << std::endl;

        continue;
      }

      if (gate_symbol == "array_index")
      {
        // array_index.2804: bits[8] = array_index(window, indices=[literal.2803], id=2804)
        size_t pos;
        pos                   = gate_args.find(", ");
        std::string symbol_in = gate_args.substr(0, pos);
        gate_args.erase(0, pos + 2);

        pos                        = gate_args.find(", ");
        std::string symbol_indices = gate_args.substr(0, pos);
        gate_args.erase(0, pos + 2);

        size_t pos_beg          = symbol_indices.find("[");
        size_t pos_end          = symbol_indices.find("]");
        std::string symbol_in_2 = symbol_indices.substr(pos_beg + 1, pos_end - pos_beg - 1);
        // std::cout << "ARRAY INDEX ... INDEX = " << symbol_in_2 << std::endl;

        // harcoded ...
        size_t sz = 1;

        auto data_in                       = logical_slices[symbol_in];
        auto data_in_2                     = logical_slices[symbol_in_2];
        logical_slices[gate_outvar_symbol] = ARRAY_INDEX(ctx, data_in, data_in_2, sz, gate_outvar_symbol);

        // std::cout << "PRODUCED DATA FOR " << gate_outvar_symbol << std::endl;
        continue;
      }

      if (gate_symbol == "not")
      {
        // not.953: bits[1] = not(bit_slice.936, id=953)
        size_t pos;
        pos                   = gate_args.find(", ");
        std::string symbol_in = gate_args.substr(0, pos);
        gate_args.erase(0, pos + 2);

        auto data_in                       = logical_slices[symbol_in];
        logical_slices[gate_outvar_symbol] = NOT(ctx, data_in, gate_outvar_symbol);

        continue;
      }

      if (gate_symbol == "concat")
      {
        //   ret concat.1269: bits[16] = concat(or.1238, or.1240, or.1242, or.1244, or.1246, or.1248, or.1250,
        //   or.1252, or.1254, or.1256, or.1258, or.1260, or.1262, or.1264, or.1266, or.1268, id=1269)
        // Remove the end ", id =.."
        size_t id_pos = gate_args.find(", id=");
        gate_args     = gate_args.substr(0, id_pos);

        std::vector<logical_slice> inputs;
        size_t pos;

        while (true)
        {
          pos = gate_args.find(", ");
          if (pos == std::string::npos)
          {
            break;
          }

          std::string symbol = gate_args.substr(0, pos);
          inputs.push_back(logical_slices[symbol]);
          gate_args.erase(0, pos + 2);

          // std::cout << "CONCAT ARG = " << symbol << std::endl;
        }

        size_t sz                          = 1;
        logical_slices[gate_outvar_symbol] = CONCAT(ctx, sz, inputs, gate_outvar_symbol);
        continue;
      }

      std::cout << "UNRECOGNIZED GATE !" << std::endl;
      abort();
    }
  }

  auto output_data = logical_slices[output_data_symbol];

  ctx.finalize();
}

int main(int argc, char** argv)
{
  /* Until we find a simple and "safe" way to pass a file to the test, we
   * consider this is not an error ...
   * One possible approach would be to convert a default .ir file to a large
   * static data processed in the test suite ?
   */
  if (argc < 2)
  {
    fprintf(stderr, "This test needs an input file, skipping.\n");
    return 0;
  }
  run<stream_ctx>(argv[1]);
  run<graph_ctx>(argv[1]);
}
