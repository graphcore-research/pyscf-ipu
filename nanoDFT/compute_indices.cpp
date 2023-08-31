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

int max_val(uint32_t val) {
    float x_candidate_f = (-1 + sqrt(1 + 8 * static_cast<float>(val))) / 2; 
    uint32_t x_candidate = static_cast<uint32_t>(floor(x_candidate_f));
    uint32_t x = ((x_candidate * (x_candidate + 1)) / 2 <= val) ? x_candidate : x_candidate - 1;
    return x;
}

typedef struct {
    uint32_t i;
    uint32_t j;
} ij_pair;


ij_pair get_i_j(uint32_t val) {
    ij_pair result;
    result.i = max_val(val);
    result.j = val - result.i * (result.i + 1) / 2;
    return result;
}


class SymmetryIndices : public Vertex {
public:
  InOut<Vector<uint32_t>>  value;
  Input<Vector<uint32_t>>  symmetry;
  Input<Vector<uint32_t>>  input_N;

  Input<Vector<uint32_t>> start;
  Input<Vector<uint32_t>> stop; 

  Output<Vector<uint32_t>> out; 

  bool compute() {

    uint32_t N = input_N.data()[0];
    
    for (uint32_t iteration = start.data()[0]; iteration < stop.data()[0]; iteration++){
      
    //for (unsigned int iteration = 0; iteration < out.size(); iteration++){
      // int ij = max_val(value.data()[iteration]);
      // int kl = value.data()[iteration] - ij*(ij+1)/2;

      // ij_pair pair_ij = get_i_j(ij);
      // ij_pair pair_kl = get_i_j(kl);
      ij_pair pair_ij = get_i_j(value.data()[2*iteration+0]);
      ij_pair pair_kl = get_i_j(value.data()[2*iteration+1]);
      uint32_t i = pair_ij.i;
      uint32_t j = pair_ij.j;
      uint32_t k = pair_kl.i;
      uint32_t l = pair_kl.j;

      switch (symmetry.data()[0]) {
        //dm_indices_func_J = lambda i,j,k,l,symmetry: jnp.array([i*N+j, j*N+i, i*N+j, j*N+i, k*N+l, l*N+k, k*N+l, l*N+k])[symmetry]
        //i*N+j, j*N+i, i*N+j, j*N+i, k*N+l, l*N+k, k*N+l, l*N+k
        case 0: { out[iteration] = i*N+j; break; }
        case 1: { out[iteration] = j*N+i; break; }
        case 2: { out[iteration] = i*N+j; break; }
        case 3: { out[iteration] = j*N+i; break; }
        case 4: { out[iteration] = k*N+l; break; }
        case 5: { out[iteration] = l*N+k; break; }
        case 6: { out[iteration] = k*N+l; break; }
        case 7: { out[iteration] = l*N+k; break; }


        //ss_indices_func_J = lambda i,j,k,l,symmetry: jnp.array([k*N+l, k*N+l, l*N+k, l*N+k, i*N+j, i*N+j, j*N+i, j*N+i])[symmetry]
        case 8: { out[iteration]  = k*N+l; break; }
        case 9: { out[iteration]  = k*N+l; break; }
        case 10: { out[iteration] = l*N+k; break; }
        case 11: { out[iteration] = l*N+k; break; }
        case 12: { out[iteration] = i*N+j; break; }
        case 13: { out[iteration] = i*N+j; break; }
        case 14: { out[iteration] = j*N+i; break; }
        case 15: { out[iteration] = j*N+i; break; }


        //dm_indices_func_K = lambda i,j,k,l,symmetry: jnp.array([k*N+j, k*N+i, l*N+j, l*N+i, i*N+l, i*N+k, j*N+l, j*N+k])[symmetry]
        case 16: { out[iteration] = k*N+j; break; }
        case 17: { out[iteration] = k*N+i; break; }
        case 18: { out[iteration] = l*N+j; break; }
        case 19: { out[iteration] = l*N+i; break; }
        case 20: { out[iteration] = i*N+l; break; }
        case 21: { out[iteration] = i*N+k; break; }
        case 22: { out[iteration] = j*N+l; break; }
        case 23: { out[iteration] = j*N+k; break; }


        //ss_indices_func_K = lambda i,j,k,l,symmetry: jnp.array([i*N+l, j*N+l, i*N+k, j*N+k, k*N+j, l*N+j, k*N+i, l*N+i])[symmetry]
        case 24: { out[iteration] = i*N+l; break; }
        case 25: { out[iteration] = j*N+l; break; }
        case 26: { out[iteration] = i*N+k; break; }
        case 27: { out[iteration] = j*N+k; break; }
        case 28: { out[iteration] = k*N+j; break; }
        case 29: { out[iteration] = l*N+j; break; }
        case 30: { out[iteration] = k*N+i; break; }
        case 31: { out[iteration] = l*N+i; break; }
      }
    }
    return true ;
  }

};

