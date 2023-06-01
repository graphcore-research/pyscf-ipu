import jax 
import jax.numpy as jnp 
from jax.config import config
import csv 
config.FLAGS.jax_platform_name = 'cpu'
# set up logging
import logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S",
        level=logging.INFO,
)

from trainer import *

from utils import set_seed
set_seed(42)

import jax
import jax.numpy as jnp
import haiku as hk
from functools import partial
import torch
from torch.utils.data import Dataset

class CharDataset(Dataset):

    def __init__(self, data, block_size, locs, energies):
        chars = sorted(list(set(data)))
        print(chars) # move _E and _P to end?
        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))
        
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data
        self.energies = energies 
        self.locs = locs
    
    def __len__(self):
        return (len(self.data) // self.block_size) #//7

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx*self.block_size: (idx+1)*self.block_size]
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        """
        arrange data and targets so that the first i elements of x
        will be asked to predict the i-th element of y. Notice that
        the eventual language model will actually make block_size
        individual predictions at the same time based on this data,
        so we are being clever and amortizing the cost of the forward
        pass of the network. So for example if block_size is 4, then
        we could e.g. sample a chunk of text "hello", the integers in
        x will correspond to "hell" and in y will be "ello". This will
        then actually "multitask" 4 separate examples at the same time
        in the language model:
        - given just "h", please predict "e" as next
        - given "he" please predict "l" next
        - given "hel" predict "l" next
        - given "hell" predict "o" next
        
        In addition, because the DataLoader will create batches of examples,
        every forward/backward pass during traning will simultaneously train
        a LOT of predictions, amortizing a lot of computation. In particular,
        for a batched input of integers X (B, T) where B is batch size and
        T is block_size and Y (B, T), the network will during training be
        simultaneously training to make B*T predictions, all at once! Of course,
        at test time we can paralellize across batch B, but unlike during training
        we cannot parallelize across the time dimension T - we have to run
        a forward pass of the network to recover the next single character of the 
        sequence along each batch dimension, and repeatedly always feed in a next
        character to get the next one.
        
        So yes there is a big asymmetry between train/test time of autoregressive
        models. During training we can go B*T at a time with every forward pass,
        but during test time we can only go B at a time, T times, with T forward 
        passes.
        """
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        locs     = torch.tensor(self.locs[idx], dtype=torch.float32)  # this has size (num_iterations)
        energies = torch.tensor(self.energies[idx], dtype=torch.float32)  # this has size (num_iterations)
        floats = torch.concatenate([locs, energies])
        return x, y, floats[:-1], floats[1:]


class CharDatasetNotAR(Dataset): # not autoregressive

    def __init__(self, data, block_size, locs, energies):
        chars = sorted(list(set(data)))
        print(chars) # move _E and _P to end?
        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))
        
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data
        self.energies = energies 
        self.locs = locs
    
    def __len__(self):
        return (len(self.data) // self.block_size) #//7

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx*self.block_size: (idx+1)*self.block_size]
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]

        x        = torch.tensor(dix, dtype=torch.long)
        locs     = torch.tensor(self.locs[idx], dtype=torch.float32)  # this has size (num_iterations)
        energies = torch.tensor(self.energies[idx], dtype=torch.float32)  # this has size (num_iterations)
        return x, locs, energies, 

# you can download this file at https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt
#text = open('input.txt', 'r').read() 


def train(self, params, opt_state=None):
        config = self.config
        lr_sheduler = lr_schedule(config, config.step_tokens if config.step_tokens is not None else self.train_dataset.block_size)
        
        optimiser = chain(
            #clip_by_global_norm(config.grad_norm_clip),
            #scale_by_adam(*config.betas),
            #add_decayed_weights(config.weight_decay, configure_decay_mask(params)),
            #scale_by_schedule(lr_sheduler),
            scale(-1), # just basic sgd?
        )
        if opt_state is None:
            opt_state = optimiser.init(params)
        #params, opt_state = map(pmap_on, (params, opt_state)) # can we use this just with pmap=1?
        loss_fn = self.hk_loss_fn.apply
            
        #@partial(jax.pmap, axis_name='num_devices') # have lr=0 parameter so we can compute validation loss
        @partial(jax.jit, donate_argnums=(0, 6), backend=args.backend)
        def update(params, subkey, 
                    x, y, 
                    floats_x, floats_y, 
                    opt_state, validation=1):
            # can compile this! 
            val, grads = jax.value_and_grad(loss_fn, has_aux=True)(params, subkey, 
                                                      x, y, 
                                                      floats_x, floats_y)

            loss = val[0]
            pred = val[1]

            #for k in grads.keys(): 
            #    grads[k] = grads[k] * validation
            grads = jax.tree_map( lambda x: x*validation, grads)

            
            updates, opt_state = optimiser.update(grads, opt_state, params)

            # this does update; very clean nice seperation! 
            params = optax.apply_updates(params, updates) # this does updates 
            return loss, pred, params, opt_state
                
        @partial(jax.pmap, axis_name='num_devices') 
        def get_loss(params, subkey, xs, ys):
            loss = loss_fn(params, subkey, xs, ys)
            return jax.lax.pmean(loss, axis_name='num_devices')
            
        def run_epoch(params, opt_state, it, split):
            is_train = split == 'train'
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)

            # we can train with 1 energy (the last one) all energies, condition on all kinds of different things! 

            losses = []
            pbar = tqdm(loader) if is_train else loader
            #jit_update = jax.jit(update,  backend="ipu")
            #jit_update = jax.jit(update, donate_argnums=(0, 6), backend="ipu")
            for batch in pbar:
                #xs, ys = map(pmap_batch, map(jnp.array, batch))
                discrete_xs, discrete_ys = jnp.array(batch[0]), jnp.array(batch[1])
                floats_xs, floats_ys     = jnp.array(batch[2]), jnp.array(batch[3])

                config.rng, subkey = jax.random.split(config.rng)

                # forward the model
                if is_train:
                    #loss, pred, params, opt_state = update(params, subkey, 
                    loss, pred, params, opt_state = update(params, subkey, 
                            discrete_xs, discrete_ys, 
                            floats_xs, floats_ys, 
                            opt_state) 


                    print(params)
                    wandb.log({"w1": np.asarray(jax.device_get(params["w1"]))})



                    loss = np.asarray(loss)#
                    pred = np.asarray(pred)

                    energy_us = (pred*std+mu)[-1]
                    energy_them = (floats_ys[0, 9*3:]*std+mu)[-1]

                    # create validaiton set defined by PCQ. 
                    if args.wandb: wandb.log({"energy_error": np.abs(energy_us - energy_them), "energy_us": np.abs(energy_us), "energy_them": np.abs(energy_them)})

                else:
                    loss = get_loss(params, subkey, xs, ys)
                    
                losses.append(loss)
                
                if is_train:
                    # report progress
                    pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss:.5f}. lr {lr_sheduler(it):e} energy {energy_us:.5f} {energy_them:.5f}")
                it += 1


                # do validation here 
                
            if not is_train:
                test_loss = float(jnp.mean(jnp.array(losses)))
                logger.info(f"test loss: {test_loss}")
                return test_loss
            
            return params, opt_state, it
        
        best_loss = float('inf')
        it = 0 # counter used for learning rate decay
        for epoch in range(config.max_epochs):
            params, opt_state, it = run_epoch(params, opt_state, it, 'train')
            if self.test_dataset is not None:
                test_loss = run_epoch(params, opt_state, 0, 'test')

            # supports early stopping based on the test loss, or just save always if no test set is provided
            good_model = self.test_dataset is None or test_loss < best_loss
            if self.config.ckpt_path is not None and good_model:
                best_loss = test_loss
                self.save_checkpoint(params, opt_state)
                
        params, opt_state = map(pmap_off, (params, opt_state))        
        return params, opt_state



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Arguments for Density Functional Theory. ')

    parser.add_argument('-wandb',     action="store_true", help='Whether to store results in wandb.') 
    parser.add_argument('-gdb',       default=9, type=int,  help='Which version of GDP to load {10, 11, 13, 17}. ')
    parser.add_argument('-prof',      action="store_true", help='Profile => stop after one iteration. ') 
    parser.add_argument('-backend',   default="cpu", type=str,  help='Backend {cpu, gpu, tpu, ipu}. ')

    args = parser.parse_args()

    if args.wandb: 
        import wandb 
        wandb.init(project="gpt")

    import pandas as pd 
    pcq  = pd.read_csv("DataFrame_QM9_PCQ_duplicates.csv")
    #text = pd.read_csv("qm9.csv", header=None, nrows=100000) 
    gdb9 = np.load("gdb9.npz", allow_pickle=True)["data"]

    #print(gdb9[:, 0])
    #exit()
    smiles = gdb9[:, 0]


    pcq_smiles = pcq["canon_smiles"].values
    pcq_smiles = dict(zip(pcq_smiles.tolist(), np.ones(pcq_smiles.shape).tolist()))

    from rdkit import Chem
    train = []
    val   = []

    for smile in tqdm(smiles): # precompute this. or perhaps use ditionary? 
        mol = Chem.MolFromSmiles(smile)
        canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=False)

        if canonical_smiles in pcq_smiles: val.append(smiles)
        else: train.append(smiles)

    print(len(train), len(val))


    #smiles         = text[0]
    #atom_txt      = "" # "".join(["".join(eval(a)) for a in text[1]]) # this is 9 or 10; perhaps try with empty token aswell. 
    #atom_positions = text[2]
    #energies       = text[3]
    #nuc            = text[4]

    from tqdm import tqdm
    energy_list = []
    smile_list  = []
    locs = []

    # GDB9; train 1M GPT on that and validate on PCQ9 ;;;; show prediction is good?
    # GDB10; train 10M GPT on that and validate on PCQ10
    # GDB11; train 25M GPT on that and validate on PCQ11
    # GDB13; train 1B GPT on that and validate on PCQ13


    # datsaet generation; generate GDB in 4 hours on pod4
    # start training GPT-small on that data. 
    # then to molecular dynamics simulation using the GPT model. 


    # intput: ("C", "N", "F", "O", "P", .., "P", "E")
    # output: ("N", "F", "O", "P", ..., "P", "E")
    # log_likihood;    atom_tokens,  positions,   energy, hlgap

    # GNN; hlgap
    #


    # yes, so atom list, atom position, energy
    # 11 discrete tokens , continuous 33 continuous ones, then finally iteration continuous ones
    # alternatively, we could do discrete for atoms and continuous for atom position? 
    # let's start with that, no smiles. 
    # perhaps store as train/val split on PCQ? 

    for i in tqdm(range(energies.shape[0])): 
        try:

            energy_list.append(np.array( [float(a) for a in energies[i].replace("\n", "").strip("[]").replace("  ", " ").split(" ")] ).reshape(1, -1))
            atom_txt += "".join(eval(text[1][i])) + "p"*(9*3) + "e"*30 #  add position and energy token
            locs.append(np.array([float(a) for a in text[2][i].replace("\n", "").strip("[]").replace("  ", " ").split(" ") if a != ""]).reshape(1, 9, 3))
        except: 
            #print(energies[i])
            pass

    print(gdb9[0])

    locs      = gdb9[:, 2:2+9*3]
    energies  = gdb9[:, 2+9*3: 2+9*3+30]
    print(locs)
    print(energies)

    mu = energies.min()
    energies = energies - mu
    std = energies.max()
    energies = energies / std
    locs = locs - locs.min()
    locs = locs / locs.max()


    train_dataset   = CharDataset(     atom_txt, 9+9*3+30, locs.reshape(-1, 9*3), energies) 
    predict_dataset = CharDatasetNotAR(atom_txt, 9+9*3+30, locs.reshape(-1, 9*3), energies)

    from model import gpt, _loss_fn, loss_fn, GPTConfig

    rng = jax.random.PRNGKey(42)
    gpt_config = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                    #n_layer=8, n_head=8, n_embd=512)
                    n_layer=2, n_head=2, n_embd=32)
                    #n_layer=4, n_head=4, n_embd=128)

    hk_loss_fn = hk.transform(partial(_loss_fn, config=gpt_config, is_training=True))


    from trainer import Trainer, TrainerConfig

    # initialize a trainer instance and kick off training
    rng, subkey = jax.random.split(rng)
    #tconf = TrainerConfig(max_epochs=2, batch_size=512//2, learning_rate=6e-4,
    tconf = TrainerConfig(max_epochs=1000, batch_size=16, learning_rate=6e-4, # use smaller batch size =O
                        lr_decay=True, warmup_tokens=512*20, 
                        final_tokens=2*len(train_dataset)*train_dataset.block_size,
                        num_workers=4, rng=subkey)
    trainer = Trainer(hk_loss_fn, train_dataset, None, tconf)

    params = trainer.init_params() 

    # define a position token  ;; then add position to this? 
    # define an energy token  ;; then add energy to this? 



    params, _ = train(trainer, params)

exit()

import jax
import jax.numpy as jnp
import haiku as hk
import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
from haiku import multinomial

# fix sampling on top of this aswell and we're done! 
#
#from utils import sample

def sample(params, model, config, x, locs, energies, steps, temperature=1.0, sample=False, top_k=None, rng=None, progress=False):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    if rng is None: 
        rng = jax.random.PRNGKey(random.randrange(2**31)) 

    block_size = config.block_size

    # sample normally the first 9 tokens; this funciton does continuous sampling.
    # we sample greedily, i.e., don't do L(0, 1)
    x_cond = x[:9+9*3] # get the first 9 atoms and the subsequent atom_positions; 


    for k in tqdm(range(steps)) if progress else range(steps):
        print(x_cond.shape, locs.shape, energies.shape)
        #x_cond = x if x.size <= block_size else x[-block_size:] # crop context if needed
        # so this should see only one e input, and then add more and more es and energie_floats! 
        logits = model(params, jnp.array(x_cond), locs, energies)
        # pluck the logits at the final step and scale by temperature
        ix = logits[-1, 5:6] #/ temperature # energy is the last one! 
        print(ix)
        # optionally crop probabilities to only the top k options
        #if top_k is not None:
        #    logits = top_k_logits(logits, top_k)
        # apply log_softmax to convert to log of probabilities (for hk.multinomial)
        #probs = jax.nn.log_softmax(logits, axis=-1)
        # sample from the distribution or take the most likely
        #if sample:
        #    rng, subkey = jax.random.split(rng)
        #    ix = hk.multinomial(subkey, probs, num_samples=1)
        #else:
        #    _, ix = jax.lax.top_k(probs, k=1)
        # append to the sequence and continue
        energies = jnp.concatenate((energies, ix), axis=0)  # almost there! 
        x = jnp.concatenate((x, jnp.ones(1)*5), axis=0) # add energy 
        exit()
    return x


data = predict_dataset 
loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=1,
                                num_workers=1)

# we will have to change sampling for this too. 
model = hk.transform(partial(gpt, config=gpt_config, is_training=False))
model = hk.without_apply_rng(model).apply
pbar = tqdm(loader) 
#jit_update = jax.jit(update, donate_argnums=(0, 4), backend="cpu")
for batch in pbar:
    #xs, ys = map(pmap_batch, map(jnp.array, batch))
    discrete_xs, xs_locs, xs_energy = np.array(batch[0][0]), np.array(batch[1][0]), np.array(batch[2][0])

    # how easy is it computing the last energy
    context = discrete_xs
    print(context)
    completion = ''.join([predict_dataset.itos[int(i)] for i in context])
    print(completion)

    # use normal sampling to generate initial atoms, then after that we can use the new type of sampling
    # for now just bypass that initial sampling and only predict atom_position / energies autoregressively
    print(discrete_xs.shape, xs_locs.shape, xs_energy.shape)
    y = sample(params, model, gpt_config, discrete_xs, xs_locs, xs_energy, 20, temperature=1.0, sample=True, top_k=10, progress=True)
    completion = ''.join([predict_dataset.itos[int(i)] for i in y])
    print(completion)

    exit()



