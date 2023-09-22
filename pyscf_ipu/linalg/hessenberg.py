import jax 
jax.config.FLAGS.jax_platform_name = 'cpu'
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import scipy

def pprint(x): print(np.around(x,3))

def cpu_our_hessenberg(A): # just change three lines of QR
    Q = jnp.identity(A.shape[0])
    R = jnp.copy(A)

    for cnt in range(1, A.shape[0] - 1):  # CHANGE: range(0, ..) to range(1, ..)
        mask = jnp.zeros(A.shape[0])
        mask = mask.at[cnt:].set(1)
        x   = R[:, cnt-1] * mask          # CHANGE: cnt to cnt-1

        u = x + jnp.concatenate([jnp.zeros(cnt), jnp.array( jnp.copysign(jnp.linalg.norm(x), -A[cnt, cnt]) ).reshape(1), jnp.zeros(R.shape[0] - 1-cnt) ])
        v = u / jnp.linalg.norm(u)

        # Householder matrix
        R = R - 2 *  v.reshape(-1, 1) @ (v.reshape(1, -1)  @ R)
        R = R - 2 *  (R @ v.reshape(-1, 1)) @ v.reshape(1, -1)  # CHANGE: add householder mult from right;;
        Q = Q - 2 * (Q @ v.reshape(-1, 1)) @ v.reshape(1, -1)

    return Q, R

def ipu_our_hessenberg(A): # just change three lines of QR
    Q = jnp.identity(A.shape[0])
    R = jnp.copy(A)


    def body(cnt, vals):
        A, R, Q = vals 
        mask = jnp.zeros(A.shape[0])
        mask = mask.at[cnt:].set(1)
        x   = R[:, cnt-1] * mask          # CHANGE: cnt to cnt-1

        u = x + jnp.concatenate([jnp.zeros(cnt), jnp.array( jnp.copysign(jnp.linalg.norm(x), -A[cnt, cnt]) 
                                                           ).reshape(1), jnp.zeros(R.shape[0] - 1-cnt) ])
        v = u / jnp.linalg.norm(u)

        # Householder matrix
        R = R - 2 *  v.reshape(-1, 1) @ (v.reshape(1, -1)  @ R)
        R = R - 2 *  (R @ v.reshape(-1, 1)) @ v.reshape(1, -1)  # CHANGE: add householder mult from right;;
        Q = Q - 2 * (Q @ v.reshape(-1, 1)) @ v.reshape(1, -1)

        return [A, R, Q]

    unroll = True
    if unroll: 
        for cnt in range(1, A.shape[0] - 1):  # CHANGE: range(0, ..) to range(1, ..)
            A, R, Q = body(cnt, [A, R, Q])
    else:
        A, R, Q = jax.lax.fori_loop(1, A.shape[0]-1, body, [A, R, Q])
        
    return Q, R


import sys 
d = int(sys.argv[1])
np.random.seed(42)
A = np.random.normal(0,1,(d,d))
A = (A + A.T)/2

Q, H = cpu_our_hessenberg(A)
_H, _Q = scipy.linalg.hessenberg(A, calc_q=True)
print("> hessenberg Q")
pprint(Q[:4, :4])
pprint(_Q[:4, :4])
print(np.max(np.abs(np.abs(Q)-np.abs(_Q))))

print("> hessenberg H")
pprint(H)
pprint(_H)
print(np.max(np.abs(np.abs(H)-np.abs(_H))))


assert np.allclose(np.abs(Q), np.abs(_Q), atol=1e-6)
assert np.allclose(np.abs(H), np.abs(_H), atol=1e-6)

jax.config.update('jax_enable_x64', False)
Q, H = jax.jit(ipu_our_hessenberg, backend="ipu")(A)
Q, H = np.asarray(Q), np.asarray(H)

print(np.max(np.abs(np.abs(Q)-np.abs(_Q))))
print(np.max(np.abs(np.abs(H)-np.abs(_H))))

assert np.allclose(np.abs(Q), np.abs(_Q), atol=1e-6)
assert np.allclose(np.abs(H), np.abs(_H), atol=1e-6)