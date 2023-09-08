#include <poplar/Vertex.hpp>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>
#include "poplar/TileConstants.hpp"

using namespace poplar;

#ifdef __IPU__
// Use the IPU intrinsics
#include <ipu_memory_intrinsics>
#include <ipu_vector_math>
#define NAMESPACE ipu
#else
// Use the std functions
#include <cmath>
#define NAMESPACE std
#endif

typedef struct {
    uint32_t i;
    uint32_t j;
} ij_pair;


ij_pair get_i_j(uint32_t val) {
    ij_pair result;
    result.i = static_cast<uint32_t>(floor((sqrt(1 + 8 * static_cast<float>(val)) - 1) / 2));
    result.j = val - result.i * (result.i + 1) / 2;
    return result;
}

void compute_symmetry(const uint32_t& ij, 
                      const uint32_t& kl, 
                      const uint32_t& N,
                      const uint32_t& symmetry, 
                      uint32_t& out) {
  ij_pair pair_ij = get_i_j(ij);
  ij_pair pair_kl = get_i_j(kl);
  uint32_t i = pair_ij.i;
  uint32_t j = pair_ij.j;
  uint32_t k = pair_kl.i;
  uint32_t l = pair_kl.j;

  switch (symmetry) {
    //dm_indices_func_J = lambda i,j,k,l,symmetry: jnp.array([i*N+j, j*N+i, i*N+j, j*N+i, k*N+l, l*N+k, k*N+l, l*N+k])[symmetry]
    //i*N+j, j*N+i, i*N+j, j*N+i, k*N+l, l*N+k, k*N+l, l*N+k
    case 0:  { out = i*N+j; break; }
    case 1:  { out = j*N+i; break; }
    case 2:  { out = i*N+j; break; }
    case 3:  { out = j*N+i; break; }
    case 4:  { out = k*N+l; break; }
    case 5:  { out = l*N+k; break; }
    case 6:  { out = k*N+l; break; }
    case 7:  { out = l*N+k; break; }


    //ss_indices_func_J = lambda i,j,k,l,symmetry: jnp.array([k*N+l, k*N+l, l*N+k, l*N+k, i*N+j, i*N+j, j*N+i, j*N+i])[symmetry]
    case 8:  { out  = k*N+l; break; }
    case 9:  { out  = k*N+l; break; }
    case 10: { out = l*N+k; break; }
    case 11: { out = l*N+k; break; }
    case 12: { out = i*N+j; break; }
    case 13: { out = i*N+j; break; }
    case 14: { out = j*N+i; break; }
    case 15: { out = j*N+i; break; }


    //dm_indices_func_K = lambda i,j,k,l,symmetry: jnp.array([k*N+j, k*N+i, l*N+j, l*N+i, i*N+l, i*N+k, j*N+l, j*N+k])[symmetry]
    case 16: { out = k*N+j; break; }
    case 17: { out = k*N+i; break; }
    case 18: { out = l*N+j; break; }
    case 19: { out = l*N+i; break; }
    case 20: { out = i*N+l; break; }
    case 21: { out = i*N+k; break; }
    case 22: { out = j*N+l; break; }
    case 23: { out = j*N+k; break; }


    //ss_indices_func_K = lambda i,j,k,l,symmetry: jnp.array([i*N+l, j*N+l, i*N+k, j*N+k, k*N+j, l*N+j, k*N+i, l*N+i])[symmetry]
    case 24: { out = i*N+l; break; }
    case 25: { out = j*N+l; break; }
    case 26: { out = i*N+k; break; }
    case 27: { out = j*N+k; break; }
    case 28: { out = k*N+j; break; }
    case 29: { out = l*N+j; break; }
    case 30: { out = k*N+i; break; }
    case 31: { out = l*N+i; break; }
  }
}


class SymmetryIndices2D : public Vertex {
public:
  // InOut<Vector<uint32_t>>  value;
  InOut<Vector<uint32_t>>  value_ij;
  InOut<Vector<uint32_t>>  value_kl;
  Input<Vector<uint32_t>>  symmetry;
  Input<Vector<uint32_t>>  input_N;

  Input<Vector<uint32_t>> start;
  Input<Vector<uint32_t>> stop; 

  Output<Vector<uint32_t>> out; 

  bool compute() {

    uint32_t N = input_N[0];
    uint32_t sym = symmetry[0];

    for (uint32_t iteration = start[0]; iteration < stop[0]; iteration++){
      // ij_pair pair_ij_kl = get_i_j(value[iteration]);
      // uint32_t ij = pair_ij_kl.i;
      // uint32_t kl = pair_ij_kl.j;
      uint32_t ij = value_ij[iteration];
      uint32_t kl = value_kl[iteration];

      compute_symmetry(ij, kl, N, sym, out[iteration]);
    }
    return true ;
  }
};

class SymmetryIndices1D : public Vertex {
public:
  InOut<Vector<uint32_t>>  value; // should be x64
  Input<Vector<uint32_t>>  symmetry;
  Input<Vector<uint32_t>>  input_N;

  Input<Vector<uint32_t>> start;
  Input<Vector<uint32_t>> stop; 

  Output<Vector<uint32_t>> out; 

  bool compute() {

    uint32_t N = input_N[0];
    uint32_t sym = symmetry[0];

    for (uint32_t iteration = start[0]; iteration < stop[0]; iteration++){
      ij_pair pair_ij_kl = get_i_j(value[iteration]); // may overflow
      uint32_t ij = pair_ij_kl.i;
      uint32_t kl = pair_ij_kl.j;

      compute_symmetry(ij, kl, N, sym, out[iteration]);
    }
    return true ;
  }
};