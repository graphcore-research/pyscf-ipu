# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import os.path as osp
import poptorch
import torch
import wandb
from data import DataConfig, create_loader, loader_info, fake_batch
from device import PopConfig, configure_poptorch
from jsonargparse import CLI
from model import ModelConfig, create_model
from torch.optim.lr_scheduler import ExponentialLR, LinearLR
from torch_geometric import seed_everything
from torchmetrics import MeanAbsoluteError
from tqdm import tqdm


def setup_run(args):
    if not args["use_wandb"]:
        return None

    run = wandb.init(
        project=args["wandb_project"],
        settings=wandb.Settings(console="wrap"),
        config=args,
    )
    name = args["data_config"].dataset
    name += f"-{tqdm.format_sizeof(args['data_config'].num_train)}"
    run.name = name + "-" + run.name
    return run


def validation(inference_model, loader):
    mae = MeanAbsoluteError()

    for data in loader:
        yhat = inference_model(data.z, data.pos, data.batch)
        mae.update(yhat, data.y[data.graphs_mask])

    return float(mae.compute())


def save_checkpoint(train_model, optimizer, lr_schedule, id="final"):
    path = osp.join(wandb.run.dir, f"checkpoint_{id}.pt")
    torch.save(
        {
            "model": train_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_schedule": lr_schedule.state_dict(),
        },
        path,
    )
    wandb.save(path, policy="now")


def train(
    seed: int = 0,
    learning_rate: float = 0.0005,
    learning_rate_decay: float = 0.96,
    update_lr_period: int = 32 * 300,
    val_period: int = 4000,
    pop_config: PopConfig = PopConfig(),
    data_config: DataConfig = DataConfig(),
    model_config: ModelConfig = ModelConfig(),
    debug: bool = False,
    wandb_project: str = "schnet-9m",
    use_wandb: bool = True,
    only_compile: bool = False,
    warmup_steps: int = 2500,
    wandb_warmup: bool = False,
):
    """
    Minimal SchNet GNN training

    Args:
        seed (int): the random number seed
        learning_rate (float): the learning rate used by the optimizer
        learning_rate_decay (float): exponential ratio to reduce the learning rate by
        update_lr_period (int): the number of steps between learning rate updates.
                                Default: 32*300*batch_size = 36M so lr is increasing 0
                                to 4M warmup, then decreasing every 36M steps
                                (1000/36 ~ 30 times)
        val_period (int): number of training steps before performing validation
        pop_config (PopConfig): configuration options for PopTorch
        data_config (DataConfig): options for data subsetting and loading
        model_config (ModelConfig): model arcitecture options
        debug (bool): enables additional logging (with perf overhead)
        use_wandb (bool): Use Weights and Biases to log benchmark results.
        only_compile (bool): Compile the and exit (no training)
        warmup_steps (int):  set to 0 to turn off. openc uses 2-3 epochs with 2-3M
                             molecules => 4-9M graphs; => steps = 10M  / batchsize
                             ~ --warmup_steps 2500
        wandb_warmup (bool): enable wandb logging of warmup steps
    """
    run = setup_run(locals())
    seed_everything(seed)
    model = create_model(model_config)
    optimizer = poptorch.optim.AdamW(model.parameters(), lr=learning_rate)
    lr_schedule = ExponentialLR(optimizer=optimizer, gamma=learning_rate_decay)

    warmup_lr_schedule = LinearLR(
        optimizer=optimizer, start_factor=0.2, total_iters=warmup_steps
    )
    train_model, inference_model = configure_poptorch(
        pop_config, debug, model, optimizer
    )

    if only_compile:
        _ = train_model(*fake_batch(model_config, train_model.options))
        return

    seed_everything(seed)
    test_loader, train_loader = create_loader(
        data_config, model_config, (inference_model.options, train_model.options)
    )

    if use_wandb:
        run.log(loader_info(test_loader, "test_loader/"))
        run.log(loader_info(train_loader, "train_loader/"))

    num_examples = 0
    bar = tqdm(total=data_config.max_num_examples, unit_scale=True)
    step = 0
    results = {}

    val_period = val_period // 10
    done = False

    while not done:
        for data in train_loader:
            if step % val_period == 0 or step < 2 or done:
                if step == 0:
                    results["num_examples"] = 0
                if step != 0:
                    train_model.detachFromDevice()
                results["val_mae"] = validation(inference_model, test_loader)
                inference_model.detachFromDevice()

                if use_wandb:
                    run.log(results)

                if done:
                    return

            if step == 10000:
                val_period = val_period * 10

            loss = train_model(data.z, data.pos, data.batch, data.y)
            num_batch_examples = int(data.graphs_mask.sum())
            num_examples += num_batch_examples
            bar.update(num_batch_examples)
            step += 1

            results["train_loss"] = float(loss.mean())
            results["num_examples"] = num_examples

            if step < warmup_steps:
                warmup_lr_schedule.step()
                train_model.setOptimizer(optimizer)
                results["lr"] = warmup_lr_schedule.get_last_lr()[0]
                if wandb_warmup:
                    wandb.log(results)  # log the entire warmup stuff!
            elif step % update_lr_period == 0:
                lr_schedule.step()
                train_model.setOptimizer(optimizer)
                results["lr"] = lr_schedule.get_last_lr()[0]

            if num_examples > data_config.max_num_examples:
                save_checkpoint(optimizer, lr_schedule, train_model)
                done = True

            bar.set_postfix(**results)


if __name__ == "__main__":
    CLI(train)
