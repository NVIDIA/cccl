#include <cub/detail/ptx-json-parser.cuh>

#include <fstream>
#include <iostream>
#include <string>

int main(int argc, char** argv)
{
  if (argc != 3)
  {
    std::cerr << "Need exactly 2 arguments.\n";
    return 1;
  }

  std::ifstream input(argv[1]);
  if (!input)
  {
    std::cerr << "Can't open the input file: " << argv[1] << "\n";
    return 1;
  }

  std::string buffer;
  input.seekg(0, std::ios::end);
  buffer.resize(input.tellg());
  input.seekg(0, std::ios::beg);
  input.read(&buffer[0], buffer.size());

  auto json = cub::detail::ptx_json::parse(argv[2], buffer);

  std::cout << json.dump() << std::endl;

  return 0;
}
