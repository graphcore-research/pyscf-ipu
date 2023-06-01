import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np 
from functools import partial
import math

class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)

class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768

normal_init = hk.initializers.RandomNormal(stddev=0.02, mean=0.0)
Linear      = partial(hk.Linear, w_init=normal_init, b_init=hk.initializers.Constant(0.0))
LayerNorm   = partial(hk.LayerNorm, axis=-1, create_scale=True, create_offset=True)

def Dropout(is_training):
    def dropout(pdrop, x):
        return hk.dropout(hk.next_rng_key(), pdrop, x) if is_training and pdrop>0.0 else x
    return dropout

def causal_self_attention(x, config, dropout):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use hk.MultiHeadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """
    assert config.n_embd % config.n_head == 0
    T, E = x.shape # Tokens, Embeddings
    LL = partial(Linear, output_size=config.n_embd)
    
    # causal mask to ensure that attention is only applied to the left in the input sequence
    mask = jnp.tril(jnp.ones((config.block_size,config.block_size)))
    
    # calculate query, key, values for all heads in batch and move head forward to be the batch dim
    k = LL(name='linear_k')(x)
    q = LL(name='linear_q')(x)
    v = LL(name='linear_v')(x)
    # nh - number of heads (n_head), hs - head size
    # (T, E=nh*hs) -> (T, nh, hs) -> (nh, T, hs)
    hs = E // config.n_head
    resh = lambda z: z.reshape(T, config.n_head, hs).swapaxes(0,1) 
    k,q,v = map(resh, (k,q,v))
    
    # causal self-attention; Self-attend: (nh, T, hs) x (nh, hs, T) -> (nh, T, T)
    att = q @ k.swapaxes(1,2) 
    att = att / math.sqrt(hs)
    att = jnp.where(mask[:T,:T], att, float('-inf'))
    
    att = jax.nn.softmax(att, axis=-1)
    att = dropout(config.attn_pdrop, att)
    # (nh, T, T) x (nh, T, hs) -> (nh, T, hs)
    y = att @ v 
    # (nh, T, hs) -> (T, nh, hs) -> (T, E) - re-assemble all head outputs
    y = y.swapaxes(0,1).reshape(T, E) 

    # output projection
    y = LL(name='linear_proj')(y)
    y = dropout(config.resid_pdrop, y)
    return y

def block(x, config, dropout):
    """ an unassuming Transformer block """
    ln1 = LayerNorm(name='layer_norm1')
    ln2 = LayerNorm(name='layer_norm2')
                
    mlp = hk.Sequential([
        Linear(4 * config.n_embd, name='linear_mlp1'),
        jax.nn.gelu,
        Linear(config.n_embd, name='linear_mlp2'),
    ])
    x = x + causal_self_attention(ln1(x), config, dropout)
    mlp = mlp(ln2(x))
    mlp = dropout(config.resid_pdrop, mlp)
    return x + mlp

def gpt(x, floats_x, config, is_training):
    """  the full GPT language model, with a context size of block_size """
    dropout = Dropout(is_training)
    # input embedding stem
    tok_emb = hk.Embed(config.vocab_size, config.n_embd, w_init=normal_init)


    pos_emb = hk.get_parameter("embeddings", shape=[config.block_size, config.n_embd], dtype=jnp.float32, init=jnp.zeros)
        
    # decoder head
    ln_f = LayerNorm(name='layer_norm_f')
    head = Linear(config.vocab_size, with_bias=False, name='linear_head')
    
    t = x.shape[0]
    assert t <= config.block_size, "Cannot forward, model block size is exhausted."

    # forward the GPT model
    token_embeddings = tok_emb(x) # each index maps to a (learnable) vector


    position_embeddings = pos_emb[:t, :] # each position maps to a (learnable) vector
    x = token_embeddings + position_embeddings # perhaps remove positional encoding here?

    #t_start = 9 + 9 * 3 + 30 # initial discrete tokens  # this gives 66, subtract one and we get 65

    #if is_training: # fix this; should be full thing, only the last token we don't give, not the last in both lcos and energy! 
    #    print(x.shape, locs.shape)
    #    x = x.at[9:9+9*3-1, :].add(locs.reshape(-1, 1)) # does this deppend on traing o sample? 
    #    x = x.at[9+9*3-1 : 9+9*3-1+30-1, :].add(E.reshape(-1, 1))

    #else: 
    x = x.at[9:, :].add(floats_x.reshape(-1, 1))
    #x = x.at[9+9*3 : 9+9*3+30, :].add(E.reshape(-1, 1))

    x = dropout(config.embd_pdrop, x)
    # transformer
    blk_fn = partial(block, config=config, dropout=dropout)
    blocks = hk.Sequential([blk_fn for _ in range(config.n_layer)])
    x = blocks(x)
    x = ln_f(x)
    return head(x)

# this is likelihood somehow. 
def cross_entropy(logits, targets):
    one_hot = jax.nn.one_hot(targets, logits.shape[-1])
    loss = -jax.nn.log_softmax(logits) * one_hot 
    loss = loss.sum() / one_hot.sum() 
    return loss 

# lets say the first ?20? tokens or so are discrete
def loss_fn(idx, targets, config, is_training):
    discrete_loss = cross_entropy(jax.vmap(gpt, in_axes=[0, None, None])(idx, config, is_training), targets)
    continuous_loss = 0 # cross_entropy(jax.vmap(gpt, in_axes=[0, None, None])(idx, config, is_training), targets)
    return discrete_loss  + continuous_loss

def _loss_fn(idx, 
             targets, 
             floats_x, 
             floats_y, 
             config, 
             is_training):

    # (16, 8, 4) # (bs, tokens, vocab)
    #print(E_x.shape, loc_x.shape, idx.shape)
    #print(idx)
    #print(_x.shape)

    out = jax.vmap(gpt, in_axes=[0, 0, None, None])(idx, floats_x, config, is_training)


    out_idx    = out[:,  :9]

    out_locs     = out[:,     9:9+9*3] 
    out_energies = out[:, 9+9*3:] 

    y_locs     = floats_y[:,       :9*3] 
    y_energies = floats_y[:, 9*3:] 
    
    # we cna fix the discrete tokens here; could also train them? a bit tricky then the model decides if it's continuous/discrete
    discrete_loss   = cross_entropy(out_idx[:, :, :4], targets[:, :out_idx.shape[1]]) 
    #continuous_loss = 0#jnp.mean(jnp.abs( _x - _y)) #0 # cross_entropy(jax.vmap(gpt, in_axes=[0, None, None])(idx, config, is_training), targets)

    #print(idx.shape, out_locs.shape, out_energies.shape, y_locs.shape, y_energies.shape)
    
    locs_loss       = jnp.mean(jnp.abs( jax.nn.sigmoid( out_locs[:, :, 4] ) - y_locs)) # mae 

    energy_predictions = jax.nn.sigmoid( out_energies[:, :, 5] )

    #E_loss          = jnp.mean(jnp.abs( jax.nn.sigmoid( out_energies[:, :, 5] ) - y_energies)) # mae
    E_loss          = jnp.mean(jnp.abs( energy_predictions - y_energies)) # mae


    #locs_loss = jnp.mean(jnp.abs(  out_locs[:, :, 4]  - loc_y)) # mae 
    #E_loss = jnp.mean(jnp.abs(  out_energy[:, :-1, 5]  - E_y)) # mae

    #print([np.asarray(a) for a in [discrete_loss, locs_loss]])

    #return discrete_loss #+ locs_loss + E_loss
    return discrete_loss + locs_loss + E_loss, energy_predictions[0]
    #return discrete_loss + locs_loss + E_loss
