"""CAD / assembly parameter decoder from JEPA + text latents.

Predicts a fixed assembly parameter vector that ``scripts/smoke_params_to_assembly``
/ ``constraints_to_geometry`` can turn into CadQuery STEP/STL. This is a
parametric assembly head — not freeform mesh generation.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

# Canonical decode slots (mm unless noted). Order is the regression target layout.
ASSEMBLY_PARAM_KEYS: tuple[str, ...] = (
    "body_radius_mm",
    "body_height_mm",
    "nose_radius_mm",
    "nose_height_mm",
    "fin_span_mm",
    "fin_thickness_mm",
    "fin_chord_mm",
    "fillet_radius_mm",
    # Vehicle-level outputs. Without these the decoder emits an airframe
    # section and nothing else, so a mission specification -- "x kg to y km" --
    # has no representable answer: the propulsion sizing and mass budget that
    # decide whether the mission closes are simply not in the output.
    "chamber_pressure_bar",
    "expansion_ratio",
    "prop_mass_kg",
    "struct_mass_kg",
    "payload_kg",
)
ASSEMBLY_PARAM_DIM = len(ASSEMBLY_PARAM_KEYS)

# Rough scales for denormalizing sigmoid outputs → mm.
_ASSEMBLY_SCALES: tuple[float, ...] = (
    80.0,   # body_radius
    400.0,  # body_height
    80.0,   # nose_radius
    150.0,  # nose_height
    100.0,  # fin_span
    10.0,   # fin_thickness
    120.0,  # fin_chord
    5.0,    # fillet
    # Scales must bracket the corpus, not the ambition. Set for orbital-class
    # vehicles (300 t propellant, 25 t payload) against a sounding-rocket
    # corpus, the mass targets normalised to 0.00002-0.00195 -- the bottom 0.2%
    # of the sigmoid's range, where the head cannot resolve them at all. A
    # 0.0076 output then denormalises to 2,281 kg for a 12 kg request. The loss
    # barely notices, because in normalised units the error is tiny.
    100.0,   # chamber_pressure_bar  (corpus 15-80)
    30.0,    # expansion_ratio       (corpus 4-25)
    500.0,   # prop_mass_kg          (corpus 16-424)
    150.0,   # struct_mass_kg        (corpus 3-117)
    60.0,    # payload_kg            (corpus 0.5-46)
)


def assembly_params_to_constraints(params_mm: dict[str, float]) -> dict[str, Any]:
    return {
        "family": "rocket_stack_proxy",
        "body_radius_mm": float(params_mm.get("body_radius_mm", 40.0)),
        "body_height_mm": float(params_mm.get("body_height_mm", 200.0)),
        "nose_radius_mm": float(params_mm.get("nose_radius_mm", 40.0)),
        "nose_height_mm": float(params_mm.get("nose_height_mm", 60.0)),
        "fin_span_mm": float(params_mm.get("fin_span_mm", 50.0)),
        "fin_thickness_mm": float(params_mm.get("fin_thickness_mm", 3.0)),
        "fin_chord_mm": float(params_mm.get("fin_chord_mm", 50.0)),
        "fillet_radius_mm": float(params_mm.get("fillet_radius_mm", 1.0)),
        # Vehicle-level fields pass through untouched. The CAD path ignores
        # them -- geometry is built from the airframe dimensions above -- but
        # they are part of the design the decoder is asked to produce, so they
        # must survive the round trip to be scored against.
        "chamber_pressure_bar": float(params_mm.get("chamber_pressure_bar", 60.0)),
        "expansion_ratio": float(params_mm.get("expansion_ratio", 20.0)),
        "prop_mass_kg": float(params_mm.get("prop_mass_kg", 5_000.0)),
        "struct_mass_kg": float(params_mm.get("struct_mass_kg", 800.0)),
        "payload_kg": float(params_mm.get("payload_kg", 200.0)),
        "material": "Al6061",
    }


def constraints_to_target_tensor(constraints: dict[str, Any]) -> torch.Tensor:
    """Normalize constraints into [0,1] regression targets."""
    scales = torch.tensor(_ASSEMBLY_SCALES, dtype=torch.float32)
    vals = []
    for key, scale in zip(ASSEMBLY_PARAM_KEYS, _ASSEMBLY_SCALES):
        raw = float(constraints.get(key, scale * 0.5))
        vals.append(max(0.0, min(1.0, raw / scale)))
    return torch.tensor(vals, dtype=torch.float32)


def tensor_to_params_mm(pred01: torch.Tensor) -> dict[str, float]:
    scales = torch.tensor(_ASSEMBLY_SCALES, dtype=pred01.dtype, device=pred01.device)
    mm = (pred01.detach().clamp(0, 1) * scales).cpu().tolist()
    if isinstance(mm, float):
        mm = [mm]
    return {k: float(v) for k, v in zip(ASSEMBLY_PARAM_KEYS, mm)}


class CadAssemblyDecoder(nn.Module):
    """MLP head: concat(geom_latent, text_latent) → assembly params in [0,1]."""

    def __init__(self, embed_dim: int = 128, hidden_dim: int | None = None, param_dim: int = ASSEMBLY_PARAM_DIM):
        super().__init__()
        hidden = hidden_dim or embed_dim * 2
        self.param_dim = param_dim
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, param_dim),
            nn.Sigmoid(),
        )

    def forward(self, geom_latent: torch.Tensor, text_latent: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([geom_latent, text_latent], dim=-1))

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "CadAssemblyDecoder":
        m = cfg.get("model", {})
        d = m.get("cad_decoder", {})
        return cls(
            embed_dim=int(m.get("embed_dim", 128)),
            hidden_dim=d.get("hidden_dim"),
            param_dim=int(d.get("param_dim", ASSEMBLY_PARAM_DIM)),
        )
