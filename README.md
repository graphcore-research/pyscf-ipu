# JaxDFT 

```
pip install -r requirements.txt
python density_functional_theory.py -id 0

         HeHe(4, 16304)
[-154.60150444 -154.86206846 -154.86206846 -154.86206846 -154.86206846
 -154.86206846 -154.86206846 -154.86206846 -154.86206846 -154.86206846
 -154.86206846 -154.86206846 -154.86206846 -154.86206846 -154.86206846
 -154.86206846 -154.86206846 -154.86206846 -154.86206846 -154.86206846
 -154.86206846 -154.86206846 -154.86206846 -154.86206846 -154.86206846
 -154.86206846 -154.86206846 -154.86206846 -154.86206846 -154.86206846
 -154.86206846 -154.86206846 -154.86206846 -154.86206846 -154.86206846
 -154.86206846 -154.86206846 -154.86206846 -154.86206846 -154.86206846
 -154.86206846 -154.86206846 -154.86206846 -154.86206846 -154.86206846
 -154.86206846 -154.86206846 -154.86206846 -154.86206846 -154.86206846]
[ pyscf ]  1.781175spyscf:              -154.862068
us:             -154.862068
diff:             0.000001
chemAcc:          0.043000
chemAcc/diff:   64129.188860
(dft) (3.1.0+1205_tf2) alexm@neverland-poplar-17:~/jaxdft-pre-experimental$ python density_functional_theory.py -id 0 -float32 -backend ipu

         HeHe(4, 16304)
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 19925.43it/s]
Compiling module jit_do_compute.2:
[##################################################] 100% Compilation Finished [Elapsed: 00:01:40.6]
[-154.60144 -154.86201 -154.86201 -154.86201 -154.86201 -154.862
 -154.86201 -154.86201 -154.86201 -154.86201 -154.86201 -154.86201
 -154.86201 -154.86201 -154.86201 -154.86201 -154.86201 -154.86201
 -154.86201 -154.86201 -154.86201 -154.86201 -154.86201 -154.86201
 -154.86201 -154.86201 -154.86201 -154.86201 -154.86201 -154.86201
 -154.86201 -154.86201 -154.86201 -154.86201 -154.86201 -154.86201
 -154.86201 -154.86201 -154.86201 -154.86201 -154.86201 -154.86201
 -154.86201 -154.86201 -154.86201 -154.86201 -154.86201 -154.86201
 -154.86201 -154.86201]
[ pyscf ]  3.896877spyscf:              -154.862068
us:             -154.862023
diff:             0.000044
chemAcc:          0.043000
chemAcc/diff:   966.459860
```

# Known bugs


- Numerical Stability: varies with {runs, devices, basis sets, forloop/jax.lax.foriloop, matmul/einsum, eigh/generalized_eigh}. Strategy: save ALL tensors from {float32, float64} through entire computational graph to wandb and compares. Find place that drifts, fix, repeat. Benchmark improvmeents in numerical precision over time. 

```
> python density_functional_theory.py -water 5 -basis "sto-3g"  -backend gpu -float32 -forloop 
...
pyscf:          -9490.940541
us:             -9490.937383
diff:             0.003158
chemAcc:          0.043000
chemAcc/diff:    13.616232      # varies, also got 200 by fix: mf_diis_space=9->8, matmuls->einsums, remove np.float32 from inv/cholesky


> python density_functional_theory.py -water 5 -basis "sto-3g" -backend gpu 
pyscf:          -9490.940541
us:             -9490.940537
diff:             0.000004
chemAcc:          0.043000
chemAcc/diff:   10093.283724
```



# TODO
[ ] build {float32, float64} numerical stability tool 

[ ] refactor tests to `test/` with pytest 

[ ] add (c) notice for {libcint, pyscf, ..} to satisfy licenses {apache 2.0, BSD-2-clause, ...}

[ ] b3lyp 

  [ ] refactor into a single file 
  
  [ ] refactor tests that product .jpg into test/
  
  [ ] write wandb tool for numerical performance 
  
[ ] write generate functionality into 'density_functional_theory.py' that does forloop; redo timing of "other matrices" in 6-31G* only did sto-3g

[ ] electron repulsion 

  [ ] fix bug in direct.py
  
  [ ] refactor functions to be callable from outside 
  
  
  [ ] inv_permutation thingy
  
[ ] gradients 






              
