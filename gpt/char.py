from jax.config import config
import time 
import csv 
config.FLAGS.jax_platform_name = 'cpu'
import logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S",
        level=logging.INFO,
)
# make deterministic
from utils import set_seed
set_seed(42)
import jax
import jax.numpy as jnp
import haiku as hk
from functools import partial
import torch
from torch.utils.data import Dataset
print(jax.default_backend(), jax.device_count(), jax.local_devices())

class CharDataset(Dataset):

    def __init__(self, data, block_size):
        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))
        
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data
    
    def __len__(self):
        return (len(self.data) - self.block_size) #//7

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.block_size + 1]
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
        return x, y
# you can download this file at https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt
text = open('input.txt', 'r').read() 
train_dataset = CharDataset(text, block_size = 128) # one line of poem is roughly 50 characters
from model import gpt, loss_fn, GPTConfig

print("asd")
rng = jax.random.PRNGKey(42)
print("asd")
gpt_config = GPTConfig(train_dataset.vocab_size, train_dataset.block_size, n_layer=8, n_head=8, n_embd=512)
print("asd")
hk_loss_fn = hk.transform(partial(loss_fn, config=gpt_config, is_training=True))
print("asd")
from trainer import Trainer, TrainerConfig

print("asd")


# initialize a trainer instance and kick off training
rng, subkey = jax.random.split(rng)
print("asd")
tconf = TrainerConfig(max_epochs=2, batch_size=512//2, learning_rate=6e-4,
                      lr_decay=True, warmup_tokens=512*20, 
                      final_tokens=2*len(train_dataset)*train_dataset.block_size,
                      num_workers=4, rng=subkey)
print("asd")
trainer = Trainer(hk_loss_fn, train_dataset, None, tconf)
print("init params")
t0 =time.time()
params = trainer.init_params() 
print("DONE: ", time.time()-t0)

params, _ = trainer.train(params)

# alright, let's sample some character-level Shakespeare
from mingpt.utils import sample
model = hk.transform(partial(gpt, config=gpt_config, is_training=False))
model = hk.without_apply_rng(model).apply
context = "O God, O God!"
x = jnp.array([train_dataset.stoi[s] for s in context])
y = sample(params, model, gpt_config, x, 2000, temperature=1.0, sample=True, top_k=10, progress=True)
completion = ''.join([train_dataset.itos[int(i)] for i in y])
print(completion)
