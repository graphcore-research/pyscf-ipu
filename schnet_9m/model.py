from typing import Optional, List
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.nn.models import schnet as pyg_schnet
from torch_geometric.nn import to_fixed_size
from dataclasses import dataclass
from device import recomputation_checkpoint


@dataclass
class ModelConfig:
    num_features: int = 1024
    num_filters: int = 256
    num_interactions: int = 5
    num_gaussians: int = 200
    k: int = 28
    cutoff: float = 15.0
    batch_size: int = 8
    use_half: bool = False
    recomputation_blocks: Optional[List[int]] = None


class TrainingModule(nn.Module):
    def __init__(self, model: nn.Module):
        """
        Wrapper that evaluates the forward pass of SchNet followed by MAE loss
        """
        super().__init__()
        self.model = model

    def forward(self, z, pos, batch, target=None):
        prediction = self.model(z, pos, batch).view(-1)

        # slice off the padding molecule
        prediction = prediction[0:-1]

        if not self.training:
            return prediction

        # Calculate MAE loss after slicing off padding molecule
        target = target[0:-1]
        return F.l1_loss(prediction, target)


class FastShiftedSoftplus(nn.Module):
    def __init__(self, needs_cast):
        """
        ShiftedSoftplus without the conditional used in native PyTorch softplus
        """
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()
        self.needs_cast = needs_cast

    def forward(self, x):
        x = x.float() if self.needs_cast else x
        u = torch.log1p(torch.exp(-x.abs()))
        v = torch.clamp_min(x, 0.0)
        out = u + v - self.shift
        out = out.half() if self.needs_cast else out
        return out

    @staticmethod
    def replace_activation(module: torch.nn.Module, needs_cast: bool):
        """
        recursively find and replace instances of default ShiftedSoftplus
        """
        for name, child in module.named_children():
            if isinstance(child, pyg_schnet.ShiftedSoftplus):
                setattr(module, name, FastShiftedSoftplus(needs_cast))
            else:
                FastShiftedSoftplus.replace_activation(child, needs_cast)


class KNNInteractionGraph(torch.nn.Module):
    def __init__(self, k: int, cutoff: float = 10.0):
        super().__init__()
        self.k = k
        self.cutoff = cutoff

    def forward(self, pos: torch.Tensor, batch: torch.Tensor):
        """
        k-nearest neighbors without dynamic tensor shapes

        :param pos (Tensor): Coordinates of each atom with shape
            [num_atoms, 3].
        :param batch (LongTensor): Batch indices assigning each atom to
                a separate molecule with shape [num_atoms]

        This method calculates the full num_atoms x num_atoms pairwise distance
        matrix. Masking is used to remove:
            * self-interaction (the diagonal elements)
            * cross-terms (atoms interacting with atoms in different molecules)
            * atoms that are beyond the cutoff distance

        Finally topk is used to find the k-nearest neighbors and construct the
        edge_index and edge_weight.
        """
        pdist = F.pairwise_distance(pos[:, None], pos, eps=0)
        rows = arange_like(batch.shape[0], batch).view(-1, 1)
        cols = rows.view(1, -1)
        diag = rows == cols
        cross = batch.view(-1, 1) != batch.view(1, -1)
        outer = pdist > self.cutoff
        mask = diag | cross | outer
        pdist = pdist.masked_fill(mask, self.cutoff)
        edge_weight, indices = torch.topk(-pdist, k=self.k)
        rows = rows.expand_as(indices)
        edge_index = torch.vstack([indices.flatten(), rows.flatten()])
        return edge_index, -edge_weight.flatten()


def arange_like(n: int, ref: torch.Tensor) -> torch.Tensor:
    return torch.arange(n, device=ref.device, dtype=ref.dtype)


def create_model(config: ModelConfig) -> TrainingModule:
    model = pyg_schnet.SchNet(
        hidden_channels=config.num_features,
        num_filters=config.num_filters,
        num_interactions=config.num_interactions,
        num_gaussians=config.num_gaussians,
        cutoff=config.cutoff,
        interaction_graph=KNNInteractionGraph(config.k, config.cutoff),
    )

    model = to_fixed_size(model, config.batch_size)
    model = TrainingModule(model)
    print_model(model, config)

    FastShiftedSoftplus.replace_activation(model, config.use_half)
    if config.use_half:
        model = model.half()

    if config.recomputation_blocks is not None:
        for block_idx in config.recomputation_blocks:
            recomputation_checkpoint(model.model.interactions[block_idx])

    return model


def print_model(model: TrainingModule, config: ModelConfig):
    from torchinfo import summary

    num_nodes = 32 * (config.batch_size - 1)
    z = torch.zeros(num_nodes).long()
    pos = torch.zeros(num_nodes, 3)
    batch = torch.zeros(num_nodes).long()
    y = torch.zeros(config.batch_size)

    summary(model, input_data=[z, pos, batch, y])
