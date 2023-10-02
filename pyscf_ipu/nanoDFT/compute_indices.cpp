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

class IndicesIJKL : public Vertex {
public:
  Input<Vector<const int>> outshape;
  //Input<Vector<const int>> j_;
  //Input<Vector<const int>> k_;
  //Input<Vector<const int>> l_;
  Input<Vector<const int>> nonzero_indices;
  
  Input<Vector<const uint32_t>> sym_;
  Input<Vector<const uint32_t>> N_;

  Input<Vector<const uint32_t>> start_;
  Input<Vector<const uint32_t>> stop_;

  Output<Vector<int>> out_; 

  bool compute() {

    const uint32_t& N     = N_[0];
    const uint32_t& sym   = sym_[0];
    const uint32_t& start = start_[0];
    const uint32_t& stop  = stop_[0];

    for (uint32_t iteration = start; iteration < stop; iteration++){

      //const int& i = i_[iteration]; 
      //const int& j = j_[iteration];
      //const int& k = k_[iteration];
      //const int& l = l_[iteration];
      const int& i = nonzero_indices[iteration*4+0];
      const int& j = nonzero_indices[iteration*4+1];
      const int& k = nonzero_indices[iteration*4+2];
      const int& l = nonzero_indices[iteration*4+3];

      int& out = out_[iteration];

      //_compute_symmetry(ij, kl, N, sym, out[iteration]);
      switch (sym) {
        case 0:  { out = i*N+j; break; }
        case 1:  { out = j*N+i; break; }
        case 2:  { out = i*N+j; break; }
        case 3:  { out = j*N+i; break; }
        case 4:  { out = k*N+l; break; }
        case 5:  { out = l*N+k; break; }
        case 6:  { out = k*N+l; break; }
        case 7:  { out = l*N+k; break; }


        //ss_indices_func_J = lambda i,j,k,l,symmetry: jnp.array([k*N+l, k*N+l, l*N+k, l*N+k, i*N+j, i*N+j, j*N+i, j*N+i])[symmetry]
        case 8:  { out = k*N+l; break; }
        case 9:  { out = k*N+l; break; }
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
    return true;
  }
};