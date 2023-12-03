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
def transformer(cfg, params, x: jnp.ndarray, position: jnp.ndarray, H_core: jnp.ndarray):
    """
    cfg: Config, from transformer_init, holds hyperparameters
    params: Current transformer parameters, initialized in init
    x: 1D array of L integers, representing the input sequence
    output: L x n_vocab logits
    """
    L, = x.shape # x is just 1D. Vmap/pmap will handle batching

    embeddings = cfg.lambda_e * params.embeddings[x, :]  # L x Dm

    # Add (learned) positional encodings
    x = jnp.concatenate([embeddings[:, :-3], position], -1) 
    L, dm = x.shape

    # Apply the transformer layers
    for layer in params.layers:
        # Layer-normalize embeddings
        #t1 = vmap(standardize)(embeddings)
        t1 = elementwise_linear(layer.norm_self_attn, x)   # L x Dm

        L, Dm = t1.shape
        nheads = cfg.heads
        qkv     = linear(layer.kqv, t1)#.reshape(L, Dm, 3)
        #q, k, v = [qkv[:, :, i].reshape(nheads, L, Dm//nheads) for i in range(3)] 
        q = jnp.transpose(qkv[:, 0*Dm:1*Dm].reshape(L, nheads, Dm//nheads), (1, 0, 2))
        k = jnp.transpose(qkv[:, 1*Dm:2*Dm].reshape(L, nheads, Dm//nheads), (1, 0, 2))
        v = jnp.transpose(qkv[:, 2*Dm:3*Dm].reshape(L, nheads, Dm//nheads), (1, 0, 2))
        score = (q @ jnp.transpose(k, (0, 2, 1))) / math.sqrt(Dm)

        if layer == 0:  # doesn't look like it helps 
            score += H_core

        attn     = jax.nn.softmax(score, axis=1)
        x = x + (attn @ v).reshape(L, Dm)

        # Layer-normalize embeddings
        #t2 = vmap(standardize)(embeddings)
        t2 = elementwise_linear(layer.norm_ff, x)          # L x Dm

        # Feedforward fully connected
        t2 = linear(layer.ffn1, t2)                         # L x Dm*4
        t2 = jax.nn.gelu(t2)
        t2 = linear(layer.ffn2, t2)                         # L x Dm

        # Add this layer's contribution into embeddings
        x = x + t2

    return score #attn #linear(params.output, embeddings)                # L x n_vocab 


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
        return jax.tree_flatten(self.__dict__, lambda a: a is not self.__dict__) # only flatten one step

    @classmethod
    def tree_unflatten(cls, aux, values):
        return ParamsDict(**jax.tree_unflatten(aux, values))

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

