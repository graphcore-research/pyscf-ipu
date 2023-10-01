import os.path as osp
from tessellate_ipu import create_ipu_tile_primitive, ipu_cycle_count, tile_map, tile_put_sharded, tile_put_replicated
from functools import partial
import numpy as np
import scipy.linalg
from typing import cast, overload, Any, Literal, Optional, Union
import jax
import jax.numpy as jnp
jax.config.FLAGS.jax_platform_name = 'cpu'

@partial(jax.jit, backend="ipu")
def eigh_tridiagonal(d, e, *, select="a", select_range = None, tol= None):
  alpha, beta = jnp.asarray(d), jnp.asarray(e)
  n = alpha.shape[0]
  if n <= 1: return jnp.real(alpha)

  beta_abs = jnp.abs(beta)
  beta_sq  = jnp.square(beta)

  # Estimate the largest and smallest eigenvalues of T using the Gershgorin circle theorem.
  off_diag_abs_row_sum = jnp.concatenate( [beta_abs[:1], beta_abs[:-1] + beta_abs[1:], beta_abs[-1:]], axis=0)
  lambda_est_max = jnp.amax(alpha + off_diag_abs_row_sum)
  lambda_est_min = jnp.amin(alpha - off_diag_abs_row_sum)

  # Upper bound on 2-norm of T.
  t_norm = jnp.maximum(jnp.abs(lambda_est_min), jnp.abs(lambda_est_max))

  # Compute the smallest allowed pivot in the Sturm sequence to avoid
  # overflow.
  finfo = np.finfo(alpha.dtype)
  one = np.ones([], dtype=alpha.dtype)
  safemin = np.maximum(one / finfo.max, (one + finfo.eps) * finfo.tiny)
  pivmin = safemin * jnp.maximum(1, jnp.amax(beta_sq))
  alpha0_perturbation = jnp.square(finfo.eps * beta_abs[0])
  abs_tol = finfo.eps * t_norm
  if tol is not None:
    abs_tol = jnp.maximum(tol, abs_tol)

  # In the worst case, when the absolute tolerance is eps*lambda_est_max and
  # lambda_est_max = -lambda_est_min, we have to take as many bisection steps
  # as there are bits in the mantissa plus 1.
  # The proof is left as an exercise to the reader.
  max_it = finfo.nmant + 1
  print(max_it)

  # Might be useful to only compute the "top k electrons//2" eigenvalues. 
  target_counts = jnp.arange(n, dtype=jnp.int32)

  # Run binary search for all desired eigenvalues in parallel, starting from
  # the interval lightly wider than the estimated
  # [lambda_est_min, lambda_est_max].
  fudge = 2.1  # We widen starting interval the Gershgorin interval a bit.
  norm_slack = jnp.array(n, alpha.dtype) * fudge * finfo.eps * t_norm
  lower = lambda_est_min - norm_slack - 2 * fudge * pivmin
  upper = lambda_est_max + norm_slack + fudge * pivmin

  # Pre-broadcast the scalars used in the Sturm sequence for improved
  # performance.
  target_shape = jnp.shape(target_counts)
  lower = jnp.broadcast_to(lower, shape=target_shape)
  upper = jnp.broadcast_to(upper, shape=target_shape)
  mid = 0.5 * (upper + lower)
  pivmin = jnp.broadcast_to(pivmin, target_shape)
  alpha0_perturbation = jnp.broadcast_to(alpha0_perturbation, target_shape)

  vertex_filename  = osp.join(osp.dirname(__file__), "tridiagonal_eigh.cpp")
  grad = create_ipu_tile_primitive(
            "Sturm" ,
            "Sturm" ,
            inputs=["alpha", "beta_sq", "pivmin", "alpha0_pertubation", "x", "id", "out_shape", "lower", "mid", "upper"], 
            outputs={ "lower_out": 7, "mid_out": 8, "upper_out": 9},
            gp_filename=vertex_filename,
            perf_estimate=100,
  )

  x = mid 
  n = x.shape[0]
  tiles = tuple(range(n))
  _alpha               = tile_put_replicated(jnp.array(alpha, dtype=jnp.float32),   tiles)
  _beta_sq             = tile_put_replicated(jnp.array(beta_sq, dtype=jnp.float32),   tiles)
  _pivmin              = tile_put_replicated(jnp.array(pivmin, dtype=jnp.float32),    tiles)
  _alpha0_perturbation = tile_put_replicated(jnp.array(alpha0_perturbation, dtype=jnp.float32),   tiles)
  _id                  = tile_put_sharded(jnp.arange(len(tiles)),   tiles)
  _out_shape           = tile_put_sharded(jnp.arange(len(tiles)).astype(np.float32),   tiles)

  _lower   = tile_put_sharded(lower,   tiles)
  _mid     = tile_put_sharded(mid,   tiles)
  _upper   = tile_put_sharded(upper,   tiles)

  def body(j, args):
    i, lower, mid, upper = args
    _x     = tile_put_replicated(jnp.array(mid.array, dtype=jnp.float32),   tiles)
    lower, mid, upper = tile_map(grad, _alpha, _beta_sq, _pivmin, 
                                            _alpha0_perturbation, _x, _id, 
                                            _out_shape, lower, mid, upper)
    return i + 1, lower, mid, upper

  vals = (0, _lower, _mid, _upper)
  body = jax.jit(body, backend="ipu")
  #while cond(vals): 
  #for i in range(12):
  #  vals = body(vals)
  vals = jax.lax.fori_loop(0, max_it, body, vals)

  return vals[2].array

  
if __name__ == "__main__":
  import jax
  import jax.numpy as jnp 
  import numpy as np 
  import scipy 
  np.random.seed(42)
  jax.config.FLAGS.jax_platform_name = 'cpu'
  np.random.seed(42)


  dim = 1024
  A = np.random.normal(0, 1, (dim,dim))
  A = (A+A.T)/2
  print(A)

  D, Q = np.linalg.eigh(A)
  print(np.max(np.abs(Q @ np.diag(D) @ Q.T - A)))

  T, Q = scipy.linalg.hessenberg(A, calc_q=True)
  print(np.around(T, 2))

  d, e = np.diag(T), np.diag(T, k=1)
  print(d, e)
  w, v = scipy.linalg.eigh_tridiagonal(d, e)
  print(w.shape, v.shape)

  assert np.allclose(T @ v - v @ np.diag(w), np.zeros((dim, dim)))
  Q_ = Q @ v 
  assert np.allclose( Q_ @ np.diag(w) @ Q_.T , A)
  assert np.allclose( Q_ @ Q_.T , np.eye(dim))
  assert np.allclose(w, D)
  print("PASSED! (scipy)")


  _w= np.array(eigh_tridiagonal(d, e))
  print(w.reshape(-1)[::128])
  print(_w.reshape(-1)[::128])
  print(np.max(np.abs(w-_w)))