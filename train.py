"""Main JEPA pretraining loop (single-device or torchrun DDP)."""

from __future__ import annotations

import argparse
import contextlib
import math
import random
import time
from pathlib import Path
from typing import Any, Iterator, Literal, cast

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from data.dataset import build_dataset
from data.transforms import collate_masked_batch
from models.jepa import JEPAModel
from utils.checkpoint import load_checkpoint, prune_checkpoints, save_checkpoint
from utils.config import apply_overrides, load_yaml, seed_everything, validate_config
from utils.distributed import (
    all_reduce_mean,
    barrier,
    cleanup_distributed,
    init_distributed,
    wrap_ddp,
)
from utils.logging import MetricLogger
from utils.schedule import warmup_cosine


def resolve_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class MixedBatchSampler(Sampler[list[int]]):
    """Yield batches with a fixed real/synthetic ratio."""

    def __init__(
        self,
        real_dataset: Dataset,
        synthetic_dataset: Dataset,
        batch_size: int,
        mix_ratio: float,
        drop_last: bool = True,
    ):
        self.real_dataset = real_dataset
        self.synthetic_dataset = synthetic_dataset
        self.batch_size = batch_size
        self.mix_ratio = mix_ratio
        self.drop_last = drop_last
        self.real_batch = max(0, int(round(batch_size * mix_ratio)))
        self.synth_batch = batch_size - self.real_batch
        if self.real_batch == 0 or self.synth_batch == 0:
            self.real_batch = batch_size if mix_ratio >= 0.5 else 0
            self.synth_batch = batch_size - self.real_batch

        self.num_batches = max(len(real_dataset) // max(self.real_batch, 1), len(synthetic_dataset) // max(self.synth_batch, 1), 1)

    def __iter__(self) -> Iterator[list[int]]:
        real_indices = list(range(len(self.real_dataset)))
        synth_indices = list(range(len(self.synthetic_dataset)))
        random.shuffle(real_indices)
        random.shuffle(synth_indices)
        real_offset = synth_offset = 0

        for _ in range(self.num_batches):
            batch: list[int] = []
            if self.real_batch > 0:
                for _ in range(self.real_batch):
                    if real_offset >= len(real_indices):
                        random.shuffle(real_indices)
                        real_offset = 0
                    batch.append(real_indices[real_offset])
                    real_offset += 1
            if self.synth_batch > 0:
                for _ in range(self.synth_batch):
                    if synth_offset >= len(synth_indices):
                        random.shuffle(synth_indices)
                        synth_offset = 0
                    # tag synthetic indices with high bit offset via negative sentinel in collate
                    batch.append(-(synth_indices[synth_offset] + 1))
                    synth_offset += 1
            yield batch

    def __len__(self) -> int:
        return self.num_batches


class MixedBatchDataset(Dataset):
    """Wrapper that resolves mixed index tags from MixedBatchSampler."""

    def __init__(self, real_dataset: Dataset, synthetic_dataset: Dataset):
        self.real_dataset = real_dataset
        self.synthetic_dataset = synthetic_dataset

    def __len__(self) -> int:
        return len(self.real_dataset) + len(self.synthetic_dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0:
            return self.synthetic_dataset[-(index + 1)]
        return self.real_dataset[index]


def build_dataloader(cfg: dict[str, Any], data_source: str) -> DataLoader:
    data_cfg = cfg["data"]
    batch_size = cfg["train"]["batch_size"]

    if data_source == "mixed":
        from data.dataset import CADSimulationDataset

        synth_cfg = {
            "num_points": data_cfg["num_points"],
            "num_fields": data_cfg["num_fields"],
            **cfg.get("synthetic", {}),
        }
        real_ds = CADSimulationDataset(
            data_dir=data_cfg["data_dir"],
            synthetic=False,
            shard_format=data_cfg.get("shard_format", "npz"),
        )
        synth_ds = CADSimulationDataset(synthetic=True, synthetic_cfg=synth_cfg, num_synthetic=256)
        mixed_ds = MixedBatchDataset(real_ds, synth_ds)
        sampler = MixedBatchSampler(
            real_ds, synth_ds, batch_size=batch_size, mix_ratio=float(data_cfg.get("mix_ratio", 0.7))
        )

        def collate(idxs: list[int]) -> dict[str, torch.Tensor]:
            batch = [mixed_ds[i] for i in idxs]
            return collate_masked_batch(batch, cfg["masking"])

        return DataLoader(
            mixed_ds,
            batch_sampler=sampler,
            collate_fn=collate,
            num_workers=data_cfg.get("num_workers", 0),
            pin_memory=data_cfg.get("pin_memory", False),
        )

    dataset = build_dataset(data_source, cfg)  # type: ignore[arg-type]
    dataset_len = len(cast(Any, dataset))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=dataset_len >= batch_size,
        num_workers=data_cfg.get("num_workers", 0),
        pin_memory=data_cfg.get("pin_memory", False),
        collate_fn=lambda batch: collate_masked_batch(batch, cfg["masking"]),
    )


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: dict[str, Any], total_steps: int):
    warmup = int(cfg["train"]["warmup_steps"])
    return warmup_cosine(optimizer, total_steps=total_steps, warmup_steps=warmup, min_lr_ratio=0.0)


def resolve_precision(cfg: dict[str, Any], device: torch.device) -> tuple[torch.dtype | None, bool, str]:
    precision = str(cfg["train"].get("precision", "auto")).lower()
    if precision == "auto" and bool(cfg["train"].get("amp", False)):
        precision = "fp16"
    if precision not in {"auto", "fp32", "fp16", "bf16"}:
        raise ValueError(f"Unsupported train.precision: {precision}")

    if device.type != "cuda":
        return None, False, "fp32"

    if precision == "fp32":
        return None, False, "fp32"
    if precision == "fp16":
        return torch.float16, True, "fp16"
    if precision == "bf16":
        return torch.bfloat16, False, "bf16"

    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, False, "bf16"
    return torch.float16, True, "fp16"


def train_loop(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    validate_config(cfg)
    seed_everything(int(cfg["train"].get("seed", 42)), bool(cfg["train"].get("deterministic", False)))
    dist_info = init_distributed(cfg["train"].get("device", "auto"))
    device = dist_info.device
    if dist_info.is_primary:
        print(f"Using device: {device} (world_size={dist_info.world_size})")

    dataloader = build_dataloader(cfg, args.data_source)
    raw_model = JEPAModel.from_config(cfg).to(device)
    if dist_info.is_primary:
        print(f"Trainable parameters: {count_parameters(raw_model):,}")
    model = wrap_ddp(raw_model, dist_info)
    jepa = raw_model  # EMA updates always on the unwrapped module

    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    max_steps = args.max_steps or cfg["train"].get("max_steps") or cfg["train"]["num_epochs"] * len(dataloader)
    scheduler = build_scheduler(optimizer, cfg, total_steps=max_steps)
    accum_steps = max(1, int(args.grad_accum_steps or cfg["train"].get("grad_accum_steps", 1)))

    step = 0
    epoch = 0
    if args.resume:
        payload = load_checkpoint(args.resume, raw_model, optimizer, scheduler, device=device)
        step = int(payload.get("step", 0))
        epoch = int(payload.get("epoch", 0))
        if dist_info.is_primary:
            print(f"Resumed from {args.resume} at step={step}")

    logger = MetricLogger(cfg["logging"]["log_dir"], cfg["logging"]["experiment_name"]) if dist_info.is_primary else None
    checkpoint_dir = Path(cfg["checkpoint"]["checkpoint_dir"])
    collapse_threshold = float(cfg["train"]["collapse_std_threshold"])
    log_every = int(cfg["train"]["log_every"])
    ckpt_every = int(cfg["train"]["checkpoint_every"])
    grad_clip = float(cfg["train"]["grad_clip"])

    precision_dtype, use_scaler, precision_mode = resolve_precision(cfg, device)
    if dist_info.is_primary:
        print(f"Precision mode: {precision_mode}")
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    model.train()
    start_time = time.perf_counter()
    data_iter = iter(dataloader)
    micro = 0
    optimizer.zero_grad(set_to_none=True)

    try:
        while step < max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                epoch += 1
                data_iter = iter(dataloader)
                batch = next(data_iter)

            points = batch["points"].to(device)
            fields = batch["fields"].to(device)
            context_mask = batch["context_mask"].to(device)
            target_masks = batch["target_masks"].to(device)
            target_block_ids = batch["target_block_ids"].to(device)

            precision_context = (
                torch.autocast(device_type=device.type, dtype=precision_dtype)
                if precision_dtype is not None
                else contextlib.nullcontext()
            )
            with precision_context:
                out = model(points, fields, context_mask, target_masks, target_block_ids)
                loss = out["loss"] / accum_steps
            scaler.scale(loss).backward()
            micro += 1

            if micro % accum_steps != 0:
                continue

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            jepa.update_target_encoder()
            scheduler.step()

            step += 1
            reduced_loss = all_reduce_mean(out["loss"].detach(), dist_info)

            if dist_info.is_primary and (step % log_every == 0 or step == 1):
                with torch.no_grad():
                    target_emb = out["target"]
                    embed_norm = target_emb.norm(dim=-1).mean().item()
                    embed_std = target_emb.std(dim=0).mean().item()

                if embed_std < collapse_threshold and logger is not None:
                    logger.warn(
                        f"Possible representation collapse at step {step}: "
                        f"target embedding std={embed_std:.6f} < {collapse_threshold}"
                    )

                elapsed = time.perf_counter() - start_time
                samples_per_sec = (step * cfg["train"]["batch_size"] * accum_steps * dist_info.world_size) / max(
                    elapsed, 1e-6
                )
                if logger is not None:
                    logger.log(
                        step,
                        {
                            "loss": float(reduced_loss.item()),
                            "lr": scheduler.get_last_lr()[0],
                            "grad_norm": float(grad_norm),
                            "embed_norm": embed_norm,
                            "embed_std": embed_std,
                            "samples_per_sec": samples_per_sec,
                            "epoch": epoch,
                            "world_size": dist_info.world_size,
                            "grad_accum_steps": accum_steps,
                        },
                    )

            if dist_info.is_primary and step % ckpt_every == 0:
                ckpt_path = checkpoint_dir / f"step_{step:06d}.pt"
                save_checkpoint(ckpt_path, raw_model, optimizer, scheduler, step, epoch, cfg)
                prune_checkpoints(checkpoint_dir, keep_last=cfg["checkpoint"]["keep_last"])

        barrier(dist_info)
        if dist_info.is_primary:
            final_path = checkpoint_dir / "latest.pt"
            save_checkpoint(final_path, raw_model, optimizer, scheduler, step, epoch, cfg)
            print(f"Training finished at step {step}. Checkpoint: {final_path}")
    finally:
        cleanup_distributed()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="JEPA-CAD pretraining")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    parser.add_argument(
        "--set",
        type=str,
        action="append",
        default=None,
        help="Override config keys, e.g. --set train.lr=3e-4 --set masking.grid_size=[6,6,6]",
    )
    parser.add_argument(
        "--data-source",
        type=str,
        choices=["real", "synthetic", "mixed"],
        default="synthetic",
        help="Training data source",
    )
    parser.add_argument("--max-steps", type=int, default=None, help="Stop after N optimizer steps")
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=None,
        help="Gradient accumulation steps (also usable under torchrun DDP)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    cfg = apply_overrides(cfg, args.set)
    train_loop(cfg, args)


if __name__ == "__main__":
    main()
