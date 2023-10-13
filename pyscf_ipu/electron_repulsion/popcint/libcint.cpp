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

#include "libcint.c"


class Grad : public Vertex {
public:
  // TODO: Change InOut to Input. 
  // Using InOut so it's float* instead of const float* (which would require changing 30k lines in libcint.c)
  InOut<Vector<float>> mat;
  InOut<Vector<int>> shls_slice;
  InOut<Vector<int>> ao_loc;
  InOut<Vector<int>> atm;
  InOut<Vector<int>> bas;
  InOut<Vector<float>> env;
  Input<Vector<int>> natm;
  Input<Vector<int>> nbas;
  Input<Vector<int>> which_integral;
  Output<Vector<float>> out; 

  bool compute() {
        float * _env = env.data();
        int  *_bas = bas.data(); 
        int  *_atm = atm.data(); 
        int  *_shls_slice = shls_slice.data(); 
        int  *_ao_loc = ao_loc.data(); 
        float * _mat = mat.data(); 

        if (which_integral.data()[0] == INT1E_KIN){
          GTOint2c(
            (int (*)(dtype *out, FINT *dims, FINT *shls, FINT *atm, FINT natm, FINT *bas, FINT nbas, dtype *env, CINTOpt *opt, dtype *cache))
          int1e_kin_sph, out.data(), 1, 0, _shls_slice, _ao_loc, NULL, _atm, natm.data()[0], _bas,  nbas.data()[0], _env);
        }
        else if (which_integral.data()[0] == INT1E_NUC){
          GTOint2c(
            (int (*)(dtype *out, FINT *dims, FINT *shls, FINT *atm, FINT natm, FINT *bas, FINT nbas, dtype *env, CINTOpt *opt, dtype *cache))
            int1e_nuc_sph, out.data(), 1, 0, _shls_slice, _ao_loc, NULL, _atm, natm.data()[0], _bas,  nbas.data()[0], _env);
        }
        else if (which_integral.data()[0] == INT1E_OVLP){
          GTOint2c(
            (int (*)(dtype *out, FINT *dims, FINT *shls, FINT *atm, FINT natm, FINT *bas, FINT nbas, dtype *env, CINTOpt *opt, dtype *cache))
            int1e_ovlp_sph, out.data(), 1, 0, _shls_slice, _ao_loc, NULL, _atm, natm.data()[0], _bas,  nbas.data()[0], _env);
        }
       if (which_integral.data()[0] == INT1E_OVLP_IP){
          GTOint2c(
            (int (*)(dtype *out, FINT *dims, FINT *shls, FINT *atm, FINT natm, FINT *bas, FINT nbas, dtype *env, CINTOpt *opt, dtype *cache))
              int1e_ipovlp_sph, out.data(), 3, 0, _shls_slice, _ao_loc, NULL, _atm, natm.data()[0], _bas,  nbas.data()[0], _env);
        }
        else if (which_integral.data()[0] == INT1E_KIN_IP){
          GTOint2c(
            (int (*)(dtype *out, FINT *dims, FINT *shls, FINT *atm, FINT natm, FINT *bas, FINT nbas, dtype *env, CINTOpt *opt, dtype *cache))
              int1e_ipkin_sph, out.data(), 3, 0, _shls_slice, _ao_loc, NULL, _atm, natm.data()[0], _bas,  nbas.data()[0], _env);
        }
        else if (which_integral.data()[0] == INT1E_NUC_IP){
          GTOint2c(
            (int (*)(dtype *out, FINT *dims, FINT *shls, FINT *atm, FINT natm, FINT *bas, FINT nbas, dtype *env, CINTOpt *opt, dtype *cache))
          int1e_ipnuc_sph, out.data(), 3, 0, _shls_slice, _ao_loc, NULL, _atm, natm.data()[0], _bas,  nbas.data()[0], _env);
        }
        
 
    
    return true;
  }
};





class Int2e : public Vertex {
public:
  //"mat", "shls_slice", "ao_loc", "atm", "bas", "env"
  InOut<Vector<float>> mat;
  InOut<Vector<int>> shls_slice;
  InOut<Vector<int>> ao_loc;
  InOut<Vector<int>> atm;
  InOut<Vector<int>> bas;
  InOut<Vector<float>> env;
  Input<Vector<int>> natm;
  Input<Vector<int>> nbas;
  Input<Vector<int>> which_integral;
  Input<Vector<int>> comp;

  Output<Vector<float>> out; 

  bool compute() {
      float * _env       = env.data();
      int   *_bas        = bas.data(); 
      int   *_atm        = atm.data(); 
      int   *_shls_slice = shls_slice.data(); 
      int   *_ao_loc     = ao_loc.data(); 
      float * _mat       = mat.data(); 
     
      GTOnr2e_fill_drv(
                         (int (*)(...))int2e_sph, 
                         (void (*)(...))GTOnr2e_fill_s1,
                         NULL, 
                        out.data(), comp.data()[0], _shls_slice, _ao_loc, NULL, 
                        _atm, natm.data()[0],  
                        _bas, nbas.data()[0], 
                        _env, which_integral.data()[0]
                        );
        
        
    return true;
  }
};



class Int2e_shell : public Vertex {
public:
  //"mat", "shls_slice", "ao_loc", "atm", "bas", "env"
  InOut<Vector<float>> mat;
  InOut<Vector<int>> shls_slice;
  InOut<Vector<int>> ao_loc;
  InOut<Vector<int>> atm;
  InOut<Vector<int>> bas;
  InOut<Vector<float>> env;
  Input<Vector<int>> natm;
  Input<Vector<int>> nbas;
  Input<Vector<int>> which_integral;
  Input<Vector<int>> comp;
  Input<Vector<int>> i; 
  Input<Vector<int>> j; 

  Output<Vector<float>> out; 

  bool compute() {
      float * _env       = env.data();
      int   *_bas        = bas.data(); 
      int   *_atm        = atm.data(); 
      int   *_shls_slice = shls_slice.data(); 
      int   *_ao_loc     = ao_loc.data(); 
      float * _mat       = mat.data(); 
     
      WHICH_INTEGRAL = which_integral.data()[0]; 
      dtype buf[312];  

      GTOnr2e_fill_s1(  (int (*)(...))int2e_sph,  NULL, out.data(), buf, comp.data()[0], i.data()[0], j.data()[0], 
                        _shls_slice, _ao_loc, NULL, _atm, natm.data()[0], _bas, nbas.data()[0], _env, which_integral.data()[0]);
        
    return true;
  }
};
