"""Distributed training helpers (DDP) with single-process fallback."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist


@dataclass(frozen=True, slots=True)
class DistInfo:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_primary(self) -> bool:
        return self.rank == 0


def init_distributed(device_pref: str = "auto") -> DistInfo:
    """Initialize process group when launched under torchrun / env://."""

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    enabled = world_size > 1

    if device_pref == "cpu":
        device = torch.device("cpu")
    elif device_pref == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda", local_rank if enabled else 0)
    elif device_pref == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda", local_rank if enabled else 0)
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_pref)

    if enabled and not dist.is_initialized():
        backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        if device.type == "cuda":
            torch.cuda.set_device(device)

    return DistInfo(
        enabled=enabled,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
    )


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def wrap_ddp(model: torch.nn.Module, info: DistInfo) -> torch.nn.Module:
    if not info.enabled:
        return model
    from torch.nn.parallel import DistributedDataParallel as DDP

    if info.device.type == "cuda":
        return DDP(model, device_ids=[info.local_rank], output_device=info.local_rank)
    return DDP(model)


def all_reduce_mean(value: torch.Tensor, info: DistInfo) -> torch.Tensor:
    if not info.enabled:
        return value
    tensor = value.detach().clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor / info.world_size


def barrier(info: DistInfo) -> None:
    if info.enabled:
        dist.barrier()
