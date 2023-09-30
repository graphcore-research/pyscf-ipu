# popcint
Libcint (manually) compiled to IPU implementing 

```
import pyscf
mol = pyscf.gto.Mole([["H", (0,0,0)], ["H", (0,0,1)]], basis="sto3g")
mol.build()
mol.intor("int1e_nuc")  # nuclear integral 
mol.intor("int1e_kin")  # kinetic integral 
mol.intor("int1e_ovlp") # overlap integral 

mol.intor("int1e_ipnuc")  # gradient of nuclear integral 
mol.intor("int1e_ipkin")  # gradient of kinetic integral 
mol.intor("int1e_ipovlp") # gradient of overlap integral 

mol.intor("int2e_sph")     # electron repulsion integral 
mol.intor("int2e_ip1_sph") # gradient (ip1) of electron repulsion integral 
```

You can test all integrals with `./cpp_libcint.sh -all`. The C++ plumbing to run all integrals is in place and all (but kinetic) pass a simple H2 test-case in STO3G (this compiles and runs libcint.c both with CPU/G++ and IPU/tesselate).  

```
> ./cpp_libcint.sh -all

Compiling with C++
Done compiling. Calling C code from python. 
[N=2]

[Nuclear Integral]
CPU:     2.763163926555734e-07
Compiling module jit_ipu_intor1e.0:
[##################################################] 100% Compilation Finished [Elapsed: 00:00:15.6]
IPU:     2.763163926555734e-07

[Kinetic Integral]
CPU:     -1.8022852765753328e-08
Compiling module jit_ipu_intor1e.1:
[##################################################] 100% Compilation Finished [Elapsed: 00:00:15.9]
IPU:     -1.05722721688295e-08

[Overlap Integral]
CPU:     -1.2445099606406274e-07
Compiling module jit_ipu_intor1e.2:
[##################################################] 100% Compilation Finished [Elapsed: 00:00:16.0]
IPU:     -6.484635128867211e-08

[Grad Nuclear]
CPU:     7.246001532124069e-08
Compiling module jit_ipu_intor1e.3:
[##################################################] 100% Compilation Finished [Elapsed: 00:00:15.8]
IPU:     7.246001532124069e-08

[Grad Kinetic]
CPU:     0.22741775584087665
[ 0.         -0.         -0.          0.          0.         -0.
 -0.          0.          0.          0.19630939 -0.19630939  0.        ]
[-0.0000000e+00  5.6303712e-04 -5.6303712e-04 -0.0000000e+00
 -1.4645594e-01  2.4947241e-02 -2.2242269e-02  1.6426709e-01
 -0.0000000e+00 -3.1108368e-02 -1.6386819e-01  9.0011621e-01]
Compiling module jit_ipu_intor1e.4:
[##################################################] 100% Compilation Finished [Elapsed: 00:00:15.8]
T[1.2]: inside CINTset_pairdata
T[1.2]: inside CINTset_pairdata
IPU:     0.1963094174861908
[ 0.         -0.         -0.          0.          0.         -0.
 -0.          0.          0.          0.19630939 -0.19630939  0.        ]
[-0.         -0.         -0.         -0.         -0.         -0.
 -0.         -0.         -0.19630942 -0.         -0.         -0.        ]

[Grad Overlap]
CPU:     6.077975783780332e-08
Compiling module jit_ipu_intor1e.5:
[##################################################] 100% Compilation Finished [Elapsed: 00:00:15.5]
IPU:     6.077975783780332e-08

[Electron Repulsion Integral]
CPU:     -4.443460513425812e-08
Compiling module jit_ipu_getints4c.6:
[##################################################] 100% Compilation Finished [Elapsed: 00:00:15.8]
IPU:     -2.953344391265489e-08

[Grad of Electron Repulsion Integral]
CPU:     1.341920186359591e-07
Compiling module jit_ipu_getints4c.7:
[##################################################] 100% Compilation Finished [Elapsed: 00:00:15.6]
IPU:     1.1929085744211143e-07
```
