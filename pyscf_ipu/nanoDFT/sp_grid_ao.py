import jax
import jax.numpy as jnp
import jax.random as random

def dense_sp_tensor(shape, sparsity, key):
    total_elements = jnp.prod(jnp.array(shape))
    num_zeros = int(sparsity * total_elements)
    
    flat_tensor = jnp.concatenate([jnp.zeros(num_zeros), jnp.ones(total_elements - num_zeros)])
    shuffled_tensor = flat_tensor[random.permutation(key, flat_tensor.shape[0], independent=True)]    
    return shuffled_tensor.reshape(shape)

def main():
    # consts
    jax.config.FLAGS.jax_platform_name = 'ipu'
    key = random.PRNGKey(42)
    key, subkey = random.split(key)

    def _sp_mm(A, B, shape):
        idxs, vals = A
        rows, cols = idxs[0], idxs[1]
        prod = B.take(cols, axis=0)*jnp.expand_dims(vals, axis=-1)
        return jax.ops.segment_sum(prod, rows, shape)

    # A - grid_AO, B - dense_matrix
    # dense
    A = dense_sp_tensor((3, 11, 7), 0.5, key) # use diff primes

    # # simulate fully zero out cols
    A = A.at[2,5,:].set(0.0)
    A = A.at[3,8,:].set(0.0)

    B = random.uniform(subkey, (7, 7)) + 1e-10
    # sparse
    A_idxs = jnp.nonzero(A)
    A_vals = A[A_idxs] # (100,)
    A_idxs = jnp.asarray(A_idxs, dtype=jnp.int16) # (3,100)

    # grid_AO_dm = sharded_grid_AO[0] @ density_matrix
    # dense
    A0B_mm = A[0]@B
    # sparse
    s = A_idxs[0]==0
    A0_sp = (A_idxs[:,s][1:], A_vals[s]) # ( (2,27), (27,) )
    sp_A0B_mm = _sp_mm(A=A0_sp, B=B, shape=A.shape[1])
    print(f"{A0B_mm=}")
    print(f"{sp_A0B_mm=}")
    different = jnp.where(sp_A0B_mm!=A0B_mm)
    print(f"{A0B_mm[different]=}")
    print(f"{sp_A0B_mm[different]=}")
 
    # mult = grid_AO_dm * sharded_grid_AO  
    # dense
    mult = A * A0B_mm # (4, 10, 5) * (10, 5) -> (4, 10, 5)
    # sparse
    sp_mult = A_vals * sp_A0B_mm[(A_idxs[1], A_idxs[2])]
    # compare
    print(f"{sp_mult=}")
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
    # comapre
    print(f"{sp_rho=}")
    print(f"{rho=}")

if __name__ == "__main__":
    main()

    # grid_AO[np.abs(grid_AO)<1e-3] = 0
    # nonzero_indicies_grid_AO = np.nonzero(grid_AO)
    # nonzero_values_grid_AO   = grid_AO[nonzero_indicies_grid_AO].astype(np.float32)
    # nonzero_indicies_grid_AO = jnp.asarray(nonzero_indicies_grid_AO, dtype=np.uint16).T
    # sparse_grid_AO           = [nonzero_indicies_grid_AO, nonzero_values_grid_AO]
    # tensors.append(sparse_grid_AO)