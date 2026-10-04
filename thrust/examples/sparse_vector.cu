#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>
#include <thrust/merge.h>
#include <thrust/reduce.h>

#include <cassert>
#include <iostream>

template <typename IndexVector, typename ValueVector>
void print_sparse_vector(const IndexVector& A_index, const ValueVector& A_value)
{
  assert(A_index.size() == A_value.size());

  for (size_t i = 0; i < A_index.size(); i++)
  {
    std::cout << "(" << A_index[i] << "," << A_value[i] << ") ";
  }
  std::cout << '\n';
}

template <typename IndexVector1,
          typename ValueVector1,
          typename IndexVector2,
          typename ValueVector2,
          typename IndexVector3,
          typename ValueVector3>
void sum_sparse_vectors(
  const IndexVector1& A_index,
  const ValueVector1& A_value,
  const IndexVector2& B_index,
  const ValueVector2& B_value,
  IndexVector3& C_index,
  ValueVector3& C_value)
{
  using IndexType = typename IndexVector3::value_type;
  using ValueType = typename ValueVector3::value_type;

  assert(A_index.size() == A_value.size());
  assert(B_index.size() == B_value.size());

  const size_t A_size = A_index.size();
  const size_t B_size = B_index.size();

  // allocate storage for the combined contents of sparse vectors A and B
  IndexVector3 temp_index(A_size + B_size);
  ValueVector3 temp_value(A_size + B_size);

  // merge A and B by index
  thrust::merge_by_key(
    A_index.begin(),
    A_index.end(),
    B_index.begin(),
    B_index.end(),
    A_value.begin(),
    B_value.begin(),
    temp_index.begin(),
    temp_value.begin());

  // compute number of unique indices
  const size_t C_size =
    thrust::inner_product(
      temp_index.begin(),
      temp_index.end() - 1,
      temp_index.begin() + 1,
      size_t(0),
      cuda::std::plus<size_t>(),
      cuda::std::not_equal_to<IndexType>())
    + 1;

  // allocate space for output
  C_index.resize(C_size);
  C_value.resize(C_size);

  // sum values with the same index
  thrust::reduce_by_key(
    temp_index.begin(),
    temp_index.end(),
    temp_value.begin(),
    C_index.begin(),
    C_value.begin(),
    cuda::std::equal_to<IndexType>(),
    cuda::std::plus<ValueType>());
}

int main()
{
  // initialize sparse vector A with 4 elements
  const thrust::device_vector<int> A_index{2, 3, 5, 8};
  const thrust::device_vector<float> A_value{10.0f, 60.0f, 20.0f, 40.0f};

  // initialize sparse vector B with 6 elements
  const thrust::device_vector<int> B_index{1, 2, 4, 5, 7, 8};
  const thrust::device_vector<float> B_value{50.0f, 30.0f, 80.0f, 30.0f, 90.0f, 10.0f};

  // compute sparse vector C = A + B
  thrust::device_vector<int> C_index;
  thrust::device_vector<float> C_value;

  sum_sparse_vectors(A_index, A_value, B_index, B_value, C_index, C_value);

  std::cout << "Computing C = A + B for sparse vectors A and B" << '\n';
  std::cout << "A ";
  print_sparse_vector(A_index, A_value);
  std::cout << "B ";
  print_sparse_vector(B_index, B_value);
  std::cout << "C ";
  print_sparse_vector(C_index, C_value);
}
