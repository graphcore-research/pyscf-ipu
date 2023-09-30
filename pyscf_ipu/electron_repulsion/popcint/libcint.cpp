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
  //"mat", "shls_slice", "ao_loc", "atm", "bas", "env"
  Input<Vector<float>> mat;
  Input<Vector<int>> shls_slice;
  Input<Vector<int>> ao_loc;
  Input<Vector<int>> atm;
  Input<Vector<int>> bas;
  Input<Vector<float>> env;
  Input<Vector<int>> natm;
  Input<Vector<int>> nbas;
  Input<Vector<int>> which_integral;

  Output<Vector<float>> out; 

  bool compute() {
        float _env[200];
        int   _bas[200];
        int   _atm[200];
        int   _shls_slice[200];
        int   _ao_loc[200];
        float _mat[mat.size()];

        for (int i = 0; i < 200; i++){
          _env[i]=0;
          _bas[i]=0;
          _atm[i]=0;
          _shls_slice[i]=0;
          _ao_loc[i]=0;
        }

        for (int i = 0; i < env.size(); i++){ _env[i] = env.data()[i]; }
        for (int i = 0; i < bas.size(); i++){ _bas[i] = bas.data()[i]; }
        for (int i = 0; i < atm.size(); i++){ _atm[i] = atm.data()[i]; }
        for (int i = 0; i < shls_slice.size(); i++){ _shls_slice[i] = shls_slice.data()[i]; }
        for (int i = 0; i < ao_loc.size(); i++){ _ao_loc[i] = ao_loc.data()[i]; }


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
  Input<Vector<float>> mat;
  Input<Vector<int>> shls_slice;
  Input<Vector<int>> ao_loc;
  Input<Vector<int>> atm;
  Input<Vector<int>> bas;
  Input<Vector<float>> env;
  Input<Vector<int>> natm;
  Input<Vector<int>> nbas;
  Input<Vector<int>> which_integral;
  Input<Vector<int>> comp;

  Output<Vector<float>> out; 

  bool compute() {

        
        float _env[200];
        int   _bas[200];
        int   _atm[200];
        int   _shls_slice[200];
        int   _ao_loc[200];
        float _mat[mat.size()];

        for (int i = 0; i < 200; i++){
          _env[i]=0;
          _bas[i]=0;
          _atm[i]=0;
          _shls_slice[i]=0;
          _ao_loc[i]=0;
        }

        for (int i = 0; i < env.size(); i++){ _env[i] = env.data()[i]; }
        for (int i = 0; i < bas.size(); i++){ 
          //if (i < 10) printf("bas[%d]=%d", i, bas.data()[i]);
          _bas[i] = bas.data()[i]; 
        }
        for (int i = 0; i < atm.size(); i++){ _atm[i] = atm.data()[i]; }
        for (int i = 0; i < shls_slice.size(); i++){ _shls_slice[i] = shls_slice.data()[i]; }
        for (int i = 0; i < ao_loc.size(); i++){ _ao_loc[i] = ao_loc.data()[i]; }

     
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
