"""Bending modes of the assembled vehicle, free-free.

The component path already reports a first natural frequency per part, from a
CalculiX modal run on the part clamped at one end. That is a useful number about
a bracket. It is not the number a launch vehicle is designed around.

What matters is the first *elastic bending mode of the whole stack*, flying
free. Nothing clamps a rocket in flight, so the vehicle bends as a free-free
beam, and that frequency decides whether the control system can fly it: an
autopilot whose bandwidth approaches the first bending mode will drive the
structure instead of steering it, and the vehicle diverges. The usual rule is
that the first bending mode should sit several times above control bandwidth.
Slosh and actuator dynamics live in the same region and interact with it.

A per-part frequency cannot approximate this. A component clamped at one end is
a cantilever, and its frequency is dominated by an artificial boundary condition
that does not exist in flight; the assembled vehicle is longer, softer, and
bends in a shape no single part sees.

The model is an Euler-Bernoulli beam with the same running mass the flight
loads use, so both describe one vehicle rather than two. Bending stiffness comes
from the thin-shell section, EI = E pi r^3 t.

Free-free is also what makes the answer checkable. An unconstrained beam has
exactly two zero-energy modes in a plane -- translation and rotation -- and they
must come out at zero. If they do not, the stiffness matrix is wrong, and a
wrong one still returns a full set of plausible frequencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

#: A free-free uniform beam's first elastic bending mode satisfies
#: cos(bL) cosh(bL) = 1, whose first non-trivial root is this. Used to verify
#: the assembled model against the closed form.
BETA_1_L = 4.730040744862704

#: Rigid-body modes are only zero to the conditioning of the eigenproblem. This
#: is the fraction of the first elastic frequency below which a mode counts as
#: rigid; a genuine elastic mode sits orders of magnitude above it.
RIGID_BODY_TOL = 1e-3


@dataclass
class ModalResult:
    frequencies_hz: list[float]
    rigid_body_modes: int
    first_bending_hz: float
    mode_shape: list[float]
    stations_m: list[float]
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def well_posed(self) -> bool:
        """A planar free-free beam has exactly two zero-energy modes."""
        return self.rigid_body_modes == 2

    def as_dict(self) -> dict:
        return {
            "first_bending_hz": round(self.first_bending_hz, 4),
            "frequencies_hz": [round(f, 4) for f in self.frequencies_hz],
            "rigid_body_modes": self.rigid_body_modes,
            "well_posed": self.well_posed,
            "notes": list(self.notes),
        }


def beam_modes(stations: list[float], mu: list[float], ei: list[float],
               n_modes: int = 6) -> ModalResult:
    """Free-free Euler-Bernoulli bending modes of a beam.

    Two-node elements with the standard cubic shape functions and a consistent
    mass matrix. Consistent rather than lumped: lumping mass at the nodes
    discards the rotary terms and reports frequencies several percent low, which
    is the wrong direction for a margin against control bandwidth.
    """
    import numpy as np
    from scipy.linalg import eigh

    n = len(stations)
    if n < 3 or len(mu) != n or len(ei) != n:
        raise ValueError("stations, mu and ei must be equal length and >= 3")
    ndof = 2 * n
    K = np.zeros((ndof, ndof))
    M = np.zeros((ndof, ndof))

    for e in range(n - 1):
        le = stations[e + 1] - stations[e]
        if le <= 0:
            raise ValueError("stations must increase")
        # Element properties averaged over the element. The section is piecewise
        # constant, so this is exact away from a section boundary and splits the
        # difference across one.
        eie = 0.5 * (ei[e] + ei[e + 1])
        mue = 0.5 * (mu[e] + mu[e + 1])
        k = (eie / le ** 3) * np.array([
            [12.0, 6.0 * le, -12.0, 6.0 * le],
            [6.0 * le, 4.0 * le * le, -6.0 * le, 2.0 * le * le],
            [-12.0, -6.0 * le, 12.0, -6.0 * le],
            [6.0 * le, 2.0 * le * le, -6.0 * le, 4.0 * le * le]])
        m = (mue * le / 420.0) * np.array([
            [156.0, 22.0 * le, 54.0, -13.0 * le],
            [22.0 * le, 4.0 * le * le, 13.0 * le, -3.0 * le * le],
            [54.0, 13.0 * le, 156.0, -22.0 * le],
            [-13.0 * le, -3.0 * le * le, -22.0 * le, 4.0 * le * le]])
        idx = [2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3]
        K[np.ix_(idx, idx)] += k
        M[np.ix_(idx, idx)] += m

    # No boundary conditions are applied. That is the physics -- a vehicle in
    # flight is unconstrained -- and it is also the check: the two rigid-body
    # modes that result must come out at zero energy.
    vals, vecs = eigh(K, M)
    vals = np.clip(vals, 0.0, None)
    freqs = np.sqrt(vals) / (2.0 * math.pi)

    order = np.argsort(freqs)
    freqs = freqs[order]
    vecs = vecs[:, order]

    elastic = [i for i, f in enumerate(freqs) if f > 0.0]
    scale = freqs[elastic[2]] if len(elastic) > 2 else max(freqs.max(), 1.0)
    rigid = int(sum(1 for f in freqs if f < RIGID_BODY_TOL * scale))

    notes: list[str] = []
    if rigid != 2:
        notes.append(
            f"found {rigid} rigid-body modes where a free-free planar beam has "
            f"exactly two; the stiffness or mass assembly is wrong and the "
            f"elastic frequencies below cannot be trusted")

    first_idx = rigid if rigid < len(freqs) else len(freqs) - 1
    shape = vecs[0::2, first_idx]
    peak = max(abs(shape.max()), abs(shape.min()), 1e-30)
    return ModalResult(
        frequencies_hz=[float(f) for f in freqs[rigid:rigid + n_modes]],
        rigid_body_modes=rigid,
        first_bending_hz=float(freqs[first_idx]),
        mode_shape=[float(v / peak) for v in shape],
        stations_m=list(stations),
        notes=tuple(notes))


def vehicle_bending_modes(vehicle, *, youngs_pa: float, wall_m: float,
                          n_stations: int = 121, n_modes: int = 4
                          ) -> ModalResult:
    """First bending modes of the vehicle described by ``vehicle``.

    Takes the same dict ``flight_loads`` does and reuses its mass distribution,
    so a change to the stack moves both the bending moment and the frequency
    rather than only one of them.
    """
    from cadflow.flight_loads import mass_per_length

    extents = vehicle.get("section_extents")
    if not extents:
        raise ValueError(
            "vehicle has no section_extents; the mass distribution is required")
    length = float(vehicle["length_m"])
    radius = float(vehicle["radius_m"])
    if length <= 0 or radius <= 0 or wall_m <= 0:
        raise ValueError("length, radius and wall thickness must be positive")

    stations = [length * i / (n_stations - 1) for i in range(n_stations)]
    mu = mass_per_length(extents, stations)
    # A thin circular shell: I = pi r^3 t. The same section the flight-loads
    # skin stress uses, so stiffness and stress cannot describe different walls.
    inertia = math.pi * radius ** 3 * wall_m
    ei = [float(youngs_pa) * inertia] * n_stations

    # Empty stations would make the beam locally massless and send the
    # eigenproblem to infinity there. Real vehicles have no such gaps; a gap
    # means the section extents do not tile the body.
    floor = 1e-6 * max(mu)
    if min(mu) <= 0:
        mu = [max(m, floor) for m in mu]
    return beam_modes(stations, mu, ei, n_modes=n_modes)


def uniform_beam_first_mode_hz(length_m: float, youngs_pa: float,
                               inertia_m4: float, mass_per_m: float) -> float:
    """Closed form for a uniform free-free beam, for verification."""
    return (BETA_1_L ** 2) / (2.0 * math.pi * length_m ** 2) * math.sqrt(
        youngs_pa * inertia_m4 / mass_per_m)
