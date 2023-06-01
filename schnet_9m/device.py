# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from typing import Optional, List
from torch import Tensor
from torch.nn import Module
from dataclasses import dataclass
from poptorch import (
    BeginBlock,
    Options,
    OutputMode,
    recomputationCheckpoint,
    setLogLevel,
    TensorLocationSettings,
    inferenceModel,
    trainingModel,
)
from poptorch.optim import Optimizer


@dataclass
class PopConfig:
    device_iterations: int = 32
    replication_factor: int = 16
    gradient_accumulation: int = 1
    optimize_popart: bool = True
    cache_dir: str = ".poptorch_cache"
    quiet: bool = True
    offload_optimizer_state: bool = False
    pipeline_splits: Optional[List[int]] = None
    available_memory_proportion: float = 0.6
    use_stochastic_rounding: bool = True


popart_options = {
    "defaultBufferingDepth": 4,
    "accumulateOuterFragmentSettings.schedule": 2,
    "replicatedCollectivesSettings.prepareScheduleForMergingCollectives": True,
    "replicatedCollectivesSettings.mergeAllReduceCollectives": True,
}


def configure_poptorch(
    config: PopConfig, debug: bool, model: Module, optimizer: Optimizer
) -> Options:
    options = Options()
    options.outputMode(OutputMode.All)
    options.deviceIterations(config.device_iterations)
    options.replicationFactor(config.replication_factor)
    options.Training.gradientAccumulation(config.gradient_accumulation)

    if config.offload_optimizer_state:
        options.TensorLocations.setOptimizerLocation(
            TensorLocationSettings().useOnChipStorage(False)
        )

    options.Precision.enableStochasticRounding(config.use_stochastic_rounding)
    options.Precision.enableFloatingPointExceptions(debug)

    if not debug:
        options.enableExecutableCaching(config.cache_dir)

    if config.optimize_popart:
        for k, v in popart_options.items():
            options._Popart.set(k, v)

    if config.quiet and not debug:
        setLogLevel("ERR")

    num_ipus = 1

    if config.pipeline_splits is not None:
        num_ipus = len(config.pipeline_splits) + 1

    options.setAvailableMemoryProportion(
        {f"IPU{i}": config.available_memory_proportion for i in range(num_ipus)}
    )

    if config.pipeline_splits is not None:
        for index, block in enumerate(config.pipeline_splits):
            model.model.interactions[block] = BeginBlock(
                model.model.interactions[block], ipu_id=index + 1
            )

    train_model = trainingModel(model, options, optimizer)
    options = options.clone()
    options.Training.gradientAccumulation(1)
    options.deviceIterations(config.device_iterations * config.gradient_accumulation)
    inference_model = inferenceModel(model.eval(), options)
    return train_model, inference_model


def recomputation_checkpoint(module: Module):
    """Annotates the output of a module to be checkpointed instead of
    recomputed"""

    def recompute_outputs(module, inputs, outputs):
        if isinstance(outputs, Tensor):
            return recomputationCheckpoint(outputs)
        elif isinstance(outputs, tuple):
            return tuple(recomputationCheckpoint(y) for y in outputs)

    return module.register_forward_hook(recompute_outputs)
