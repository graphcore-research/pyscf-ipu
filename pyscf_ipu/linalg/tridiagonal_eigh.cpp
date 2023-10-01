#include <poplar/Vertex.hpp>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>
#include "poplar/TileConstants.hpp"
#include <print.h>

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

class Sturm : public Vertex {
public:
  Input<Vector<float>> alpha;
  Input<Vector<float>> beta_sq;
  Input<Vector<float>> pivmin;
  Input<Vector<float>> alpha0_pertubation;
  Input<Vector<float>> x;
  Input<Vector<int>> id; 
  Input<Vector<float>> out_shape; 
  Input<Vector<float>> lower; 
  Input<Vector<float>> mid;  
  Input<Vector<float>> upper; 

  Output<Vector<float>> lower_out; 
  Output<Vector<float>> mid_out;  
  Output<Vector<float>> upper_out;  

  bool compute() {
    int tile_id = id.data()[0];
    
    //q     = alpha[0] - x
    //count = jnp.where(q < 0, ones, zeros)
    //q     = jnp.where(alpha[0] == x, alpha0_perturbation, q)
    float q = alpha[0] - x.data()[tile_id];
    int count = q < 0;
    if (alpha[0] == x.data()[tile_id]) {q = alpha0_pertubation.data()[tile_id]; }

    //for i in range(1, n):
    //  q        = alpha[i] - beta_sq[i - 1] / q - x                   
    //  count    = jnp.where(q <= pivmin, count + 1, count)            
    //  q        = jnp.where(q <= pivmin, jnp.minimum(q, -pivmin), q)*/

    int n = x.size();
    float x_tile_id = x.data()[tile_id];
    float pivmin_tile_id = pivmin.data()[tile_id];
    float minus_pivmin_tile_id = -pivmin_tile_id; 

    // main bulk: takes ~ 87k cycles for 1024 matrix => ~ 80 cycles per iteration. 
    // obs: we can precompute (alpha.data()[i]-x_tile_id) using all 6 threads in parallel. 
    for (unsigned int i = 1; i < n; i++){ 
      //q        = alpha[i] - beta_sq[i - 1] / q - x 
      //q        = alpha.data()[i] - x.data()[tile_id] - beta_sq.data()[i - 1] / q ;
      q        = alpha.data()[i] - x_tile_id - beta_sq.data()[i - 1] / q ;

      //count    = jnp.where(q <= pivmin, count + 1, count)            
      //q        = jnp.where(q <= pivmin, jnp.minimum(q, -pivmin), q)
      if (q <= pivmin_tile_id){ 
        count ++;
        q = fmin(q, minus_pivmin_tile_id); 
      }
    }

    //lower  = jnp.where(counts <= target_counts, mid, lower)
    //upper  = jnp.where(counts > target_counts, mid, upper)
    //mid    = 0.5 * (lower + upper)
    int target_count = tile_id; // they are the same 
    if (count <= target_count) lower_out[0] = mid.data()[0]; 
    else {lower_out[0] = lower.data()[0]; }

    if (count > target_count) upper_out[0] = mid.data()[0]; 
    else {upper_out[0] = upper.data()[0];}

    mid_out[0] = (lower_out.data()[0] + upper_out.data()[0])/2;
      
    return true;
  }
};