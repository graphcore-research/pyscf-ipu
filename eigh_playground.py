import numpy as np
import jax
from tessellate_ipu_local import ipu_eigh
import jax.numpy as jnp

ITERATIONS = 6

A = np.random.normal(0, 1, (6,6))
A = A + A.T

vals, vects = np.linalg.eigh(A)

print("GROUND TRUTH SHAPE:", vals.shape, vects.shape, vals.reshape(6,1).shape)
# exit()

accum_errs = []
for i in range(ITERATIONS):

    us_vects, us_vals = jax.jit(ipu_eigh)(A, num_iters=i)
    us_vects = np.asarray(us_vects)
    us_vals = np.asarray(us_vals)
    err = us_vects - vects
    cumulated_err = np.sum(err)
    abs_err = np.abs(us_vects) - np.abs(vects)
    cumulated_abs_err = np.sum(np.abs(abs_err))
# print(vects)
# print(us_vects)
    # print(err)
    # print(abs_err)
    print("iter:", i, "cumulated abs err:", cumulated_abs_err, "cumulated err:", cumulated_err)
    accum_errs.append(cumulated_abs_err)

# vects, vals = jax.jit(ipu_eigh)(A, num_iters=10)

accum_errs_initialized = []
for i in range(ITERATIONS):
    us_vects, us_vals = jax.jit(ipu_eigh)(A, num_iters=i, initial_guess=(jnp.diag(vals), vects))
    us_vects = np.asarray(us_vects)
    us_vals = np.asarray(us_vals)
    # us_vects, us_vals = jax.jit(ipu_eigh)(A, num_iters=i, initial_guess=(vals.reshape(6,1), vects))

    err = us_vects - vects
    cumulated_err = np.sum(err)
    abs_err = np.abs(us_vects) - np.abs(vects)
    cumulated_abs_err = np.sum(np.abs(abs_err))
    print("iter:", i, "cumulated abs err:", cumulated_abs_err, "cumulated err:", cumulated_err)
    accum_errs_initialized.append(cumulated_abs_err)


from matplotlib import pyplot as plt

plt.figure()
plt.plot(accum_errs, label="random start")
plt.plot(accum_errs_initialized, label="initialized with linalg.eigh ground truth")
plt.xlabel("Iterations of ipu_eigh")
plt.ylabel("Accumulated err of abs(us_vects) - abs(vects)")
plt.yscale('log')
plt.legend()
plt.savefig("eigh_err.png")


# check the eigenvalues !