import jax
import numpy as np
import jax.numpy as jnp
import jax.random as random
from functools import partial

BACKEND = 'ipu'

def dense_sp_tensor(shape, sparsity, key):
    total_elements = np.prod(np.array(shape))
    num_zeros = (sparsity * total_elements).astype(np.int32)
    
    flat_tensor = np.concatenate([np.zeros(num_zeros), np.ones(total_elements - num_zeros)])
    shuffled_tensor = flat_tensor[random.permutation(key, flat_tensor.shape[0], independent=True)]    
    return shuffled_tensor.reshape(shape)


def _sp_mm(A, B, shape):
    idxs, vals = A
    rows, cols = idxs[0], idxs[1]
    prod = B.take(cols, axis=0)*jnp.expand_dims(vals, axis=-1)
    return jax.ops.segment_sum(prod, rows, shape)


@partial(jax.jit, static_argnums=(3,), backend=BACKEND)
def main(A, A_sp, B, s):
    A_idxs, A_vals = A_sp
    s = jnp.array(s)
    # grid_AO_dm = sharded_grid_AO[0] @ density_matrix
    # dense
    A0B_mm = A[0]@B
    # sparse
    filtered_idxs = jnp.take(A_idxs, s, axis=1)
    A0_idxs = jnp.take(filtered_idxs, indices=jnp.arange(1, filtered_idxs.shape[0]), axis=0)
    A0_vals = jnp.take(A_vals, s)
    sp_A0B_mm = _sp_mm(A=(A0_idxs, A0_vals), B=B, shape=A.shape[1])
    # compare (only works without jitting)
    # print(f"{A0B_mm=}")
    # print(f"{sp_A0B_mm=}")
    # different = jnp.where(sp_A0B_mm!=A0B_mm)
    # print(f"{A0B_mm[different]=}")
    # print(f"{sp_A0B_mm[different]=}")
 
    # mult = grid_AO_dm * sharded_grid_AO  
    # dense
    mult = A * A0B_mm # (3, 11, 7) * (11, 7) -> (3, 11, 7)
    # sparse
    sp_mult = A_vals * sp_A0B_mm[(A_idxs[1], A_idxs[2])]
    # compare
    print(f"{sp_mult.shape=}")
    print(f"{mult[(A_idxs[0], A_idxs[1], A_idxs[2])]=}")

    # rho = jnp.sum(mult, axis=2) 
    # dense
    rho = jnp.sum(mult, axis=2)
    # sparse sum 
    sp_rho = jax.lax.scatter_add(
        operand=jnp.zeros(shape=A.shape[:-1]), 
        scatter_indices=A_idxs[:-1].T,
        updates=sp_mult,
        dimension_numbers=jax.lax.ScatterDimensionNumbers(
            update_window_dims=(),
            inserted_window_dims=(0, 1),
            scatter_dims_to_operand_dims=(0, 1)
        )
    )
    # dense 
    rho = jnp.expand_dims(rho, axis=2) # (4, 11, 1)
    A_rho = A * rho
    sum_A_rho = jnp.sum(A_rho, axis=0) 
    # sparse
    A_sp_rho = A_vals * sp_rho[(A_idxs[0], A_idxs[1])]

    sum_A_sp_rho = jax.lax.scatter_add(
        operand=jnp.zeros(shape=A.shape[1:]), 
        scatter_indices=A_idxs[1:].T,
        updates=A_sp_rho,
        dimension_numbers=jax.lax.ScatterDimensionNumbers(
            update_window_dims=(),
            inserted_window_dims=(0, 1),
            scatter_dims_to_operand_dims=(0, 1)
        )
    )
    # dense
    V_xc = A[0].T @ sum_A_rho
    # sparse
    AO_idxs_T = jnp.vstack([A0_idxs[1], A0_idxs[0]])
    sp_V_xc = _sp_mm(A=(AO_idxs_T, A0_vals), B=sum_A_sp_rho, shape=A.shape[-1])

    return (V_xc, sp_V_xc), (sum_A_rho, sum_A_sp_rho), (rho, sp_rho)


if __name__ == "__main__":
    # consts
    jax.config.FLAGS.jax_platform_name = 'cpu'
    key = random.PRNGKey(42)
    key, subkey = random.split(key)

    # A - grid_AO, B - dense_matrix
    # dense
    A = dense_sp_tensor((4, 11, 7), 0.5, key) # use diff primes

    # simulate fully zero out cols
    A[2,5,:] = 0.0
    A[1,8,:] = 0.0

    B = random.uniform(subkey, (7, 7)) + 1e-10
    # sparse
    A_idxs = np.nonzero(A)
    A_vals = A[A_idxs] # (150,)
    A_idxs = np.asarray(A_idxs).astype(np.int32) # (4,150)
    s = np.where(A_idxs[0]==0)[0]
    s = tuple(s.tolist())

    (V_xc, sp_V_xc), (sum_A_rho, sum_A_sp_rho), (rho, sp_rho) = main(A, (jnp.asarray(A_idxs), jnp.asarray(A_vals)), B, s=s)
    
    # compare results
    # print(f"{jax.device_get(rho)=}")
    # print(f"{jax.device_get(sp_rho)=}")
    # print(np.testing.assert_almost_equal(
    #     jax.device_get(sp_rho),
    #     jax.device_get(rho),
    #     decimal=5
    # ))

    # print(f"{jax.device_get(sum_A_rho)=}")
    # print(f"{jax.device_get(sum_A_sp_rho)=}")
    # print(np.testing.assert_almost_equal(
    #     jax.device_get(sum_A_sp_rho),
    #     jax.device_get(sum_A_rho),
    #     decimal=5
    # ))

    print(f"{jax.device_get(V_xc)=}")
    print(f"{jax.device_get(sp_V_xc)=}")
    print(np.testing.assert_almost_equal(
        jax.device_get(sp_V_xc),
        jax.device_get(V_xc),
        decimal=4
    ))

