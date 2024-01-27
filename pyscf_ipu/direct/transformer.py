""" Pure-from-the-ground-up transformer, based on https://github.com/vpj/jax_transformer/blob/master/transformer.py """
import jax
from jax import vmap
import jax.numpy as jnp
from functools import partial
import jax.experimental.host_callback
import math 
import numpy as np 

def rand(rng, f, shape, **kwargs):
    rng, rng1 = jax.random.split(rng)
    return rng, f(rng1, shape, **kwargs)

def linear_init_uniform(rng: jax.random.KeyArray, in_features: int, out_features: int):
    # todo: init as kaparthy
    params = ParamsDict()
    rnd_range = 1 / in_features**0.5
    rng, params.weight = rand( rng, jax.random.uniform, (in_features, out_features), minval=-rnd_range, maxval=rnd_range,)
    params.bias = jnp.zeros((out_features,))
    return rng, params, (in_features, out_features)

def elementwise_linear_init_identity(shape): return ParamsDict(gain=jnp.ones(shape), bias=jnp.zeros(shape))

def linear(params, x: jnp.ndarray): return x @ params.weight + params.bias[None, :]

def elementwise_linear(params, x: jnp.ndarray): return params.gain[None, :] * x + params.bias[None, :]

def standardize(x, eps=1e-5): return (x - x.mean()) / (x.std() + eps)

def transformer_init(
    rng: jax.random.KeyArray,
    n_vocab: int,
    d_model: int,
    n_layers: int,
    n_heads: int,
    d_ff: int,
    max_len=4096,
):
    total_params = 0 

    # Build config struct for call
    config = ParamsDict()
    config.heads = n_heads
    if True: #flip_pe_coef():
        config.lambda_e = d_model**-0.5
        config.lambda_pe = 1.0
    else:
        config.lambda_e = d_model**-0.5
        config.lambda_pe = 1.0

    # Build initializers for params
    params = ParamsDict()

    print("_"*100)

    # Create embedding layer
    rng, params.embeddings = rand(rng, jax.random.normal, (n_vocab, d_model))
    total_params += np.prod(params.embeddings.shape)
    print("%26s %26s %26s"%("params.embeddings",params.embeddings.shape, np.prod(params.embeddings.shape)))

    rng, params.project_positions, shape = linear_init_uniform(rng, 123, d_model)
    total_params += np.prod(shape)
    print("%26s %26s %26s"%("params.project_positions",shape, np.prod(shape)))

    # For transformer layers
    params.layers = []
    for i in range(n_layers):
        layer = ParamsDict()
        layer.norm_self_attn = elementwise_linear_init_identity(d_model)
        total_params += np.prod(d_model*2)
        print("%26s %26s %26s"%("layer%i.norm_self_attn"%i, (d_model,2), np.prod((d_model, 2))))

        rng, layer.kqv, shape = linear_init_uniform(rng, d_model, d_model*3)
        total_params += np.prod(shape) # omitting bias in calculation for now
        print("%26s %26s %26s"%("layer%i.kqv"%i, shape, np.prod(shape)))

        rng, layer.c_proj, shape = linear_init_uniform(rng, d_model, d_model)
        total_params += np.prod(shape) # omitting bias in calculation for now
        print("%26s %26s %26s"%("layer%i.c_proj"%i, shape, np.prod(shape)))

        layer.norm_ff = elementwise_linear_init_identity(d_model)
        total_params += np.prod(d_model*2)
        print("%26s %26s %26s"%("layer%i.norm_ff"%i, (d_model,2), np.prod((d_model, 2))))

        rng, layer.ffn1, shape = linear_init_uniform(rng, d_model, d_ff)
        total_params += np.prod(shape)
        print("%26s %26s %26s"%("layer%i.ffn1"%i, shape, np.prod(shape))) 

        rng, layer.ffn2, shape = linear_init_uniform(rng, d_ff, d_model)
        total_params += np.prod(shape)
        print("%26s %26s %26s"%("layer%i.ffn2"%i, shape, np.prod(shape)))

        params.layers.append(layer)

    # Final normalization and output layer
    print("total: ", total_params)

    return rng, config, params, total_params


@partial(jax.jit, static_argnums=0)
def transformer(cfg, params, x: jnp.ndarray, pos: jnp.ndarray, H_core: jnp.ndarray, L_inv, dm_init,diff_JK, V_xc, H_init):
    """
    cfg: Config, from transformer_init, holds hyperparameters
    params: Current transformer parameters, initialized in init
    x: 1D array of L integers, representing the input sequence
    output: L x n_vocab logits
    """
    embeddings = cfg.lambda_e * params.embeddings[x, :]  # L x Dm
    L, Dm      = embeddings.shape
    nheads     = cfg.heads

    # todo: apply the same translation/rotation to hamiltonian. 
    # Roughly get f( {R@ri+t}_i ) = f( {r_i}_i )
    pos      = pos - jnp.mean(pos, axis=0).reshape(1, 3) # makes jnp.mean(position, axis=0) = [0,0,0]
    cov      = jnp.cov(pos.T)
    eigvects = jnp.linalg.eigh(cov)[1] 
    pos      = pos @ eigvects # makes jnp.cov(pos.T)=jnp.eye(3) 

    # Mix of sin/cos and 3d point cloud transformers. 
    pos = jnp.concatenate([pos] +  \
                                [jnp.cos(pos*f/20*2*np.pi) for f in range(20)] + \
                                [jnp.sin(pos*f/20*2*np.pi) for f in range(20)], 
                                axis=1) #(N,3) -> (N,3+60+60) = (N, 123)
    pos = linear(params.project_positions, pos)                         # L x Dm
    all_pairs_dot = pos.reshape(-1, Dm) @ pos.reshape(-1, Dm).T  # this is L x L 
    x = embeddings + pos                                                     #  L x Dm 
    
    def block(x, layer_num, layer):
        # Layer-normalize 
        t1 = vmap(standardize)(x)                           # L x Dm 
        t1 = elementwise_linear(layer.norm_self_attn, t1)   # L x Dm

        qkv     = linear(layer.kqv, t1)                     # L x 3*Dm
        q,k,v = jnp.split(qkv, 3, axis=1)                   # (L x Dm,)*3
        q = jnp.transpose(q.reshape(L, nheads, Dm//nheads), (1, 0, 2)) # nheads x L x Dm//nheads
        k = jnp.transpose(k.reshape(L, nheads, Dm//nheads), (1, 0, 2))
        v = jnp.transpose(v.reshape(L, nheads, Dm//nheads), (1, 0, 2))

        score = (q @ jnp.transpose(k, (0, 2, 1)))  
        score = score 
        score = score / math.sqrt(Dm/nheads)                # B x L x L 

        # quantum biased attention 
        if True:  
            score = score.at[:2].add( H_core / jnp.max(jnp.abs(H_core)) )
            score  = score.at[2:4].add( all_pairs_dot / jnp.max(jnp.abs(all_pairs_dot)) )
            M = L_inv @ H_core @ L_inv.T   
            score  = score.at[4:6].add(  M / jnp.max(jnp.abs(M)) )
            score = score.at[6:8].add( dm_init / jnp.max(jnp.abs(dm_init))) 
            score = score.at[8:10].add(diff_JK / jnp.max(jnp.abs(diff_JK)))
            score = score.at[10:12].add(V_xc / jnp.max(jnp.abs(V_xc)))
            score = score.at[12:14].add(H_init / jnp.max(jnp.abs(H_init)))

        attn = jax.nn.softmax(score, axis=1) 
        y = attn @ v 
        y = y.swapaxes(0,1).reshape(L, Dm)  
        y = linear(layer.c_proj, y)
        x = x + y 

        # Layer-normalize 
        t2 = vmap(standardize)(x)
        t2 = elementwise_linear(layer.norm_ff, t2)          # L x Dm

        # Feedforward fully connected
        t2 = linear(layer.ffn1, t2)                         # L x Dm*4
        t2 = jax.nn.gelu(t2)
        t2 = linear(layer.ffn2, t2)                         # L x Dm

        # Residual connection 
        x = x + t2
        return x

    # Apply all but the last transformer block. 
    # todo: cut jit time wth jax.lax.foriloop
    for layer_num, layer in enumerate(params.layers[:-1]):
        x = jax.checkpoint(block)(x, layer_num, layer)
        #x = block(x, layer_num, layer)

    layer = params.layers[-1]
    # Prediction is last attention (without nhead = 1), and q=k so score is symmetric! 
    nheads = 1 
    t1    = vmap(standardize)(x)                           # L x Dm 
    t1    = elementwise_linear(layer.norm_self_attn, t1)   # L x Dm
    qkv   = linear(layer.kqv, t1)
    q,k,v = jnp.split(qkv, 3, axis=1)
    q     = jnp.transpose(q.reshape(L, nheads, Dm//nheads), (1, 0, 2))
    k     = q 
    #v = jnp.transpose(v.reshape(L, nheads, Dm//nheads), (1, 0, 2))
    score = (q @ jnp.transpose(k, (0, 2, 1))) / math.sqrt(Dm*nheads) 
        
    M = score[0] 
    return M 

import types
import json
import jax

import numbers

def is_simple_type(x):
    return isinstance(x, (numbers.Number, bool, str))

@jax.tree_util.register_pytree_node_class
class ParamsDict(types.SimpleNamespace):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def tree_flatten(self):
        return jax.tree_util.tree_flatten(self.__dict__, lambda a: a is not self.__dict__) # only flatten one step

    @classmethod
    def tree_unflatten(cls, aux, values):
        return ParamsDict(**jax.tree_util.tree_unflatten(aux, values))

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, 
            sort_keys=True, indent=4)

    def __hash__(self):
        # Should overload setattr to warn if setattr is called after hash has been computed
        return hash(tuple(hash(x) for (_,x) in self.__dict__.items()))

    def print(self, path = ''):
        for (k,v) in self.items(path):
            print(k + ':',v)

    @classmethod
    def labels_aux(cls, path, obj):
        if isinstance(obj, (list, tuple)) and any(not is_simple_type(x) for x in obj):
            for i,vi in enumerate(obj):
                yield from cls.labels_aux(f'{path}[{i}]', vi)
        elif isinstance(obj, dict):
            for (k,v) in obj.items():
                yield from cls.labels_aux(path + '/' + k, v)
        elif isinstance(obj, ParamsDict):
            yield from cls.labels_aux(path, obj.__dict__)
        else:
            yield (path, obj)

    def items(self, path = ''):
        yield from self.labels_aux(path, self)

    def to_float32(self):
        def convert_to_float32(x):
            if isinstance(x, jnp.ndarray) and x.dtype == jnp.float64:
                return x.astype(jnp.float32)
            return x

        # Create a new ParamsDict instance with converted arrays
        new_dict = jax.tree_map(convert_to_float32, self.__dict__)
        return ParamsDict(**new_dict)
        #self.__dict__ = jax.tree_map(convert_to_float32, self.__dict__)

    def to_float64(self):
        def convert_to_float64(x):
            if isinstance(x, jnp.ndarray) and x.dtype == jnp.float32:
                return x.astype(jnp.float64)
            return x

        # Create a new ParamsDict instance with converted arrays
        new_dict = jax.tree_map(convert_to_float64, self.__dict__)
        return ParamsDict(**new_dict)
        #self.__dict__ = jax.tree_map(convert_to_float64, self.__dict__)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # DFT options 
    parser.add_argument('-basis',   type=str,   default="sto3g")  
    parser.add_argument('-level',   type=int,   default=0)

    # GD options 
    parser.add_argument('-backend', type=str,       default="cpu") 
    parser.add_argument('-lr',      type=float,     default=2.5e-4)
    parser.add_argument('-steps',   type=int,       default=100000)
    parser.add_argument('-bs',      type=int,       default=8)
    parser.add_argument('-val_bs',      type=int,   default=8)
    parser.add_argument('-mol_repeats',  type=int,  default=16) # How many time to optimize wrt each molecule. 

    # energy computation speedups 
    parser.add_argument('-foriloop',  action="store_true") # whether to use jax.lax.foriloop for sparse_symmetric_eri (faster compile time but slower training. )
    parser.add_argument('-xc_f32',   action="store_true") 
    parser.add_argument('-eri_f32',  action="store_true") 
    parser.add_argument('-eri_bs',  type=int, default=8) 

    parser.add_argument('-normal',     action="store_true") 
    parser.add_argument('-wandb',      action="store_true") 
    parser.add_argument('-prof',       action="store_true") 
    parser.add_argument('-visualize',  action="store_true") 
    parser.add_argument('-skip',       action="store_true", help="skip pyscf test case") 

    # dataset 
    parser.add_argument('-qm9',        action="store_true") 
    parser.add_argument('-benzene',        action="store_true") 
    parser.add_argument('-hydrogens',        action="store_true") 
    parser.add_argument('-water',        action="store_true") 
    parser.add_argument('-waters',        action="store_true") 
    parser.add_argument('-alanine',        action="store_true") 
    parser.add_argument('-states',         type=int,   default=1)
    parser.add_argument('-workers',        type=int,   default=5) 
    parser.add_argument('-precompute',        action="store_true")  # precompute labels; only run once for data{set/augmentation}.
        # do noise schedule, start small slowly increase 
    parser.add_argument('-wiggle_var',     type=float,   default=0.05, help="wiggle N(0, wiggle_var), bondlength=1.5/30")
    parser.add_argument('-eri_threshold',  type=float,   default=1e-10, help="loss function threshold only")
    parser.add_argument('-rotate_deg',     type=float,   default=90, help="how many degrees to rotate")

    # models 
    parser.add_argument('-nn',       action="store_true", help="train nn, defaults to GD") 
    parser.add_argument('-tiny',     action="store_true") 
    parser.add_argument('-small',    action="store_true") 
    parser.add_argument('-base',     action="store_true") 
    parser.add_argument('-medium',   action="store_true") 
    parser.add_argument('-large',    action="store_true") 
    parser.add_argument('-xlarge',   action="store_true") 
    opts = parser.parse_args()

    # initialize model 
    # transformer tiny 5M 
    d_model= 192
    n_heads = 6
    n_layers = 12

    from train import nao
    rnd_key = jax.random.PRNGKey(42)
    n_vocab = nao("C", opts.basis) + nao("N", opts.basis) + \
              nao("O", opts.basis) + nao("F", opts.basis) + \
              nao("H", opts.basis) 

    rnd_key, cfg, params, total_params = transformer_init(
        rnd_key,
        n_vocab,
        d_model =d_model,
        n_layers=n_layers,
        n_heads =n_heads,
        d_ff    =d_model*4,
    )


    # compute dummy output 
    from train import batched_state, summary
    opts.alanine = True 
    alanine = [ ["H", ( 2.000 ,  1.000,  -0.000)], ["C", ( 2.000 ,  2.090,   0.000)], ["H", ( 1.486 ,  2.454,   0.890)], ["H", ( 1.486 ,  2.454,  -0.890)],
            ["C", ( 3.427 ,  2.641,  -0.000)], ["O", ( 4.391 ,  1.877,  -0.000)], ["N", ( 3.555 ,  3.970,  -0.000)], ["H", ( 2.733 ,  4.556,  -0.000)],
            ["C", ( 4.853 ,  4.614,  -0.000)], ["H", ( 5.408 ,  4.316,   0.890)], ["C", ( 5.661 ,  4.221,  -1.232)], ["H", ( 5.123 ,  4.521,  -2.131)], 
            ["H", ( 6.630 ,  4.719,  -1.206)], ["H", ( 5.809 ,  3.141,  -1.241)], ["C", ( 4.713 ,  6.129,   0.000)], ["O", ( 3.601 ,  6.653,   0.000)],
            ["N", ( 5.846 ,  6.835,   0.000)], ["H", ( 6.737 ,  6.359,  -0.000)], ["C", ( 5.846 ,  8.284,   0.000)], ["H", ( 4.819 ,  8.648,   0.000)],
            ["H", ( 6.360 ,  8.648,   0.890)], ["H", ( 6.360 ,  8.648,  -0.890)], ]
    state = batched_state(alanine, opts, opts.bs, \
                    wiggle_num=0, do_pyscf=False, validation=False, \
                    extrapolate=False, mol_idx=0)
    summary(state)
    
    output = jax.jit(jax.vmap(transformer, in_axes=(None, None, 0, 0, 0, 0), out_axes=(0)), 
                    static_argnums=(0,),
                    backend="cpu")(cfg, \
            params, state.ao_types, state.pos.astype(jnp.float32), state.H_core.astype(jnp.float32), state.L_inv.astype(jnp.float32))

            
    print(np.sum(output)) # 162.58726108305348


    # store model 
    import pickle 
    pickle.dump(params, open("checkpoints/example.pickle", "wb")) 

    # reload model 
    new_params = pickle.load(open("checkpoints/example.pickle", "rb"))

    # check that output remains the same 
    new_output = jax.jit(jax.vmap(transformer, in_axes=(None, None, 0, 0, 0, 0), out_axes=(0)), 
                    static_argnums=(0,),
                    backend="cpu")(cfg, \
            new_params, state.ao_types, state.pos.astype(jnp.float32), state.H_core.astype(jnp.float32), state.L_inv.astype(jnp.float32))

    assert np.allclose(output, new_output)
    print("TEST CASE PASSED!")


    