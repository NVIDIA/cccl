#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/merge.h>
#include <thrust/set_operations.h>

#include <iostream>

// This example illustrates use of the set operation algorithms
//  - merge
//  - set_union
//  - set_intersection
//  - set_difference
//  - set_symmetric_difference
//
// In this context a "set" is simply a sequence of sorted values,
// allowing the standard set operations to be performed more efficiently
// than on unsorted data.  Since the output of a set operation is a valid
// set (i.e. a sorted sequence) it is possible to apply the set operations
// in a nested fashion to compute arbitrary set expressions.
//
// Set operation usage notes:
//   - The output set size is variable (except for thrust::merge),
//     so the return value is important.
//   - Generally one would conservatively allocate storage for the output
//     and then resize or shrink an output container as necessary.
//     Alternatively, one can compute the exact output size by
//     outputting to a discard_iterator.  This approach is more computationally
//     expensive (approximately 2x), but conserves memory capacity.
//     Refer to the set_intersection_size function for implementation details.
//   - Sets are allowed to have duplicate elements, which are carried
//     through to the output in a algorithm-specific manner.  Refer
//     to the full documentation for precise semantics.

// helper routine
template <typename String, typename Vector>
void print(const String& s, const Vector& v)
{
  std::cout << s << " [";
  for (const auto& e : v)
  {
    std::cout << " " << e;
  }
  std::cout << " ]" << '\n';
}

template <typename Vector>
void merge(const Vector& A, const Vector& B)
{
  // merged output is always exactly A.size() + B.size()
  Vector C(A.size() + B.size());

  thrust::merge(A.begin(), A.end(), B.begin(), B.end(), C.begin());

  print("merge(A,B)", C);
}

template <typename Vector>
void set_union(const Vector& A, const Vector& B)
{
  // union output is at most A.size() + B.size()
  Vector C(A.size() + B.size());

  // set_union returns an iterator C_end denoting the end of input
  typename Vector::iterator C_end;

  C_end = thrust::set_union(A.begin(), A.end(), B.begin(), B.end(), C.begin());

  // shrink C to exactly fit output
  C.erase(C_end, C.end());

  print("Union(A,B)", C);
}

template <typename Vector>
void set_intersection(const Vector& A, const Vector& B)
{
  // intersection output is at most min(A.size(), B.size())
  Vector C(thrust::min(A.size(), B.size()));

  // set_union returns an iterator C_end denoting the end of input
  typename Vector::iterator C_end;

  C_end = thrust::set_intersection(A.begin(), A.end(), B.begin(), B.end(), C.begin());

  // shrink C to exactly fit output
  C.erase(C_end, C.end());

  print("Intersection(A,B)", C);
}

template <typename Vector>
void set_difference(const Vector& A, const Vector& B)
{
  // difference output is at most A.size()
  Vector C(A.size());

  // set_union returns an iterator C_end denoting the end of input
  typename Vector::iterator C_end;

  C_end = thrust::set_difference(A.begin(), A.end(), B.begin(), B.end(), C.begin());

  // shrink C to exactly fit output
  C.erase(C_end, C.end());

  print("Difference(A,B)", C);
}

template <typename Vector>
void set_symmetric_difference(const Vector& A, const Vector& B)
{
  // symmetric difference output is at most A.size() + B.size()
  Vector C(A.size() + B.size());

  // set_union returns an iterator C_end denoting the end of input
  typename Vector::iterator C_end;

  C_end = thrust::set_symmetric_difference(A.begin(), A.end(), B.begin(), B.end(), C.begin());

  // shrink C to exactly fit output
  C.erase(C_end, C.end());

  print("SymmetricDifference(A,B)", C);
}

template <typename Vector>
void set_intersection_size(const Vector& A, const Vector& B)
{
  // computes the exact size of the intersection without allocating output
  const thrust::discard_iterator<> C_begin;
  const auto C_end = thrust::set_intersection(A.begin(), A.end(), B.begin(), B.end(), C_begin);

  std::cout << "set_intersection_size(A,B) " << (C_end - C_begin) << '\n';
}

int main()
{
  int a[] = {0, 2, 4, 5, 6, 8, 9};
  int b[] = {0, 1, 2, 3, 5, 7, 8};

  const thrust::device_vector<int> A(a, a + sizeof(a) / sizeof(int));
  const thrust::device_vector<int> B(b, b + sizeof(b) / sizeof(int));

  print("Set A", A);
  print("Set B", B);

  merge(A, B);
  set_union(A, B);
  set_intersection(A, B);
  set_difference(A, B);
  set_symmetric_difference(A, B);

  set_intersection_size(A, B);

  return 0;
}
