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

int max_val(int N) {
    float x_candidate_f = (-1 + sqrt(1 + 8 * (float)N)) / 2; 
    int x_candidate = (int) floor(x_candidate_f);
    int x = ((x_candidate * (x_candidate + 1)) / 2 <= N) ? x_candidate : x_candidate - 1;
    return x;
}

typedef struct {
    int32_t i;
    int32_t j;
} ij_pair;


ij_pair get_i_j(int32_t val) {
    ij_pair result;
    result.i = max_val(val);
    result.j = val - result.i * (result.i + 1) / 2;
    return result;
}


class SymmetryIndices : public Vertex {
public:
  Input<Vector<int>>  value;
  Input<Vector<int>>  symmetry;
  Input<Vector<int>>  input_N;

  Input<Vector<int>> start;
  Input<Vector<int>> stop; 

  Output<Vector<int>> out; 

  bool compute() {

    int N = input_N.data()[0];

    for (unsigned int iteration = start.data()[0]; iteration < stop.data()[0]; iteration++){
    //for (unsigned int iteration = 0; iteration < out.size(); iteration++){
      int ij = max_val(value.data()[iteration]);
      int kl = value.data()[iteration] - ij*(ij+1)/2;

      ij_pair pair_ij = get_i_j(ij);
      ij_pair pair_kl = get_i_j(kl);
      int i = pair_ij.i;
      int j = pair_ij.j;
      int k = pair_kl.i;
      int l = pair_kl.j;

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



int64_t max_val_int64(int64_t N) {
    //double x_candidate_f = (-1 + sqrt(1 + 8 * (double)N)) / 2; 
    int64_t x_candidate_f = (-1 + sqrt(1 + 8 * (int64_t)N)) / 2; 
    int64_t x_candidate = (int64_t) floor(x_candidate_f);
    int64_t x = ((x_candidate * (x_candidate + 1)) / 2 <= N) ? x_candidate : x_candidate - 1;
    return x;
}

typedef struct {
    int64_t i;
    int64_t j;
} ij_pair_int64;


ij_pair_int64 get_i_j_int64(int64_t val) {
    ij_pair_int64 result;
    result.i = max_val_int64(val);
    result.j = val - result.i * (result.i + 1) / 2;
    return result;
}


class PairSymmetryIndices : public Vertex {
public:
  Input<Vector<int>> value_low;
  Input<Vector<int>> value_high;
  Input<Vector<int>> symmetry;
  Input<Vector<int>> input_N;

  Input<Vector<int>> start;
  Input<Vector<int>> stop; 

  Output<Vector<int>> out; 

  bool compute() {
    int N = input_N.data()[0];

    for (unsigned int iteration = start.data()[0]; iteration < stop.data()[0]; iteration++){
    //for (unsigned int iteration = 0; iteration < out.size(); iteration++){
      int64_t current_val = (int64_t)value_low.data()[iteration] + 
                            //(((int64_t)value_high.data()[iteration]) << 31);
                            (int64_t)(value_high.data()[iteration] << 31);

      int64_t ij = max_val_int64(current_val);
      int64_t kl = current_val - ij*(ij+1)/2;

      ij_pair_int64 pair_ij = get_i_j_int64(ij);
      ij_pair_int64 pair_kl = get_i_j_int64(kl);
      int i = (int)pair_ij.i;
      int j = (int)pair_ij.j;
      int k = (int)pair_kl.i;
      int l = (int)pair_kl.j;

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

