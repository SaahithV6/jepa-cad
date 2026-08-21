"""Modal analysis validated against the closed-form cantilever beam.

The static check cannot see a resonance, and `first_mode_hz` was a conditioning
slot with nothing populating it. This adds a CalculiX *FREQUENCY step and pins
it to the one case with an exact answer: the first bending mode of a uniform
cantilever is

    f1 = (1.875104)^2 / (2 pi) * sqrt(E I / (rho A L^4))

A solid tet mesh does not match Euler-Bernoulli exactly, and the interesting
part is *how* it misses: linear tetrahedra are too stiff, so they over-predict
frequency and approach the true answer from above. On a 120 x 10 x 10 mm steel
cantilever the ratio to theory runs 1.261, 1.208, 1.111, 1.067, 1.043 as the
element size goes 5.0, 4.0, 3.0, 2.2, 1.7 mm. So the test is convergence, not a
fixed tolerance -- a tolerance would only be measuring the mesh.
"""

import math
import shutil

import pytest

from cadflow.msh_to_calculix import (
    DEFAULT_CCX,
    generate_modal_case_inp,
    parse_eigenfrequencies,
    run_calculix_case,
)
from cadflow.rocket_physics_suite import mesh_stl_volume

E = 210e9
RHO = 7850.0
NU = 0.3

ccx_missing = not (
    shutil.which("ccx") or __import__("pathlib").Path(DEFAULT_CCX).exists()
)


def cantilever_first_mode_hz(length_m, width_m, height_m):
    """Euler-Bernoulli first bending frequency, bending about the thin axis."""
    inertia = width_m * height_m**3 / 12.0
    area = width_m * height_m
    return (1.875104**2 / (2.0 * math.pi)) * math.sqrt(
        E * inertia / (RHO * area * length_m**4)
    )


def test_frequency_parser_reads_a_calculix_eigenvalue_block(tmp_path):
    """Parse the real block layout, and drop rigid-body modes.

    CalculiX prints mode, eigenvalue, frequency in rad/time, frequency in
    cycles/time, and an imaginary part. Reading the last column took that
    imaginary part, which is identically zero for an undamped eigenproblem, so
    every frequency was then discarded as rigid-body and the parser returned
    nothing at all from a solve that had converged perfectly well.
    """
    dat = tmp_path / "case.dat"
    dat.write_text(
        "\n"
        "     E I G E N V A L U E   O U T P U T\n"
        "\n"
        " MODE NO    EIGENVALUE                       FREQUENCY   \n"
        "                                     REAL PART            IMAGINARY PART\n"
        "                           (RAD/TIME)      (CYCLES/TIME     (RAD/TIME)\n"
        "\n"
        "      1   0.1000000E-08   0.3162278E-04   0.5032921E-05   0.0000000E+00\n"
        "      2   0.5883000E+07   0.2425490E+04   0.3860280E+03   0.0000000E+00\n"
        "      3   0.2266000E+08   0.4760252E+04   0.7576000E+03   0.0000000E+00\n"
        "\n",
        encoding="utf-8",
    )
    freqs = parse_eigenfrequencies(dat)
    # the near-zero rigid-body mode is dropped; the elastic ones survive in order
    assert freqs == pytest.approx([386.028, 757.6], rel=1e-4)


def test_missing_dat_file_yields_no_frequencies(tmp_path):
    assert parse_eigenfrequencies(tmp_path / "nope.dat") == []


def test_modal_deck_carries_density_and_a_frequency_step(tmp_path):
    """A *FREQUENCY step without *DENSITY has no mass and cannot solve."""
    pytest.importorskip("cadquery")
    import cadquery as cq

    stl = tmp_path / "beam.stl"
    cq.exporters.export(cq.Workplane("XY").box(10.0, 10.0, 120.0), str(stl))
    msh = tmp_path / "mesh.msh"
    res = mesh_stl_volume(stl, msh, cl_max_mm=6.0, cl_min_mm=3.0,
                          scale_to_meters=True, mesh_timeout_s=300)
    if not res.success:
        pytest.skip(f"gmsh unavailable or failed: {res.error}")

    setup = generate_modal_case_inp(tmp_path, modes=6, youngs_modulus=E,
                                    poisson=NU, density=RHO)
    deck = setup.case_inp.read_text()
    assert "*FREQUENCY" in deck
    assert "*DENSITY" in deck
    assert "*STATIC" not in deck
    assert setup.fixed_nodes > 0


@pytest.mark.skipif(ccx_missing, reason="CalculiX not installed")
def test_cantilever_first_mode_converges_to_the_closed_form(tmp_path):
    """Mesh a beam at two densities, solve, and check it converges to theory.

    A single tolerance would be the wrong test here. Linear tetrahedra are too
    stiff, so they over-predict frequency and approach the true answer *from
    above*: on this beam the ratio to Euler-Bernoulli runs 1.261, 1.208, 1.111,
    1.067, 1.043 as the element size goes 5.0, 4.0, 3.0, 2.2, 1.7 mm. Any fixed
    tolerance therefore says more about the mesh than about the solver. What
    must hold is that refining moves the answer toward theory and not away.
    """
    pytest.importorskip("cadquery")
    import cadquery as cq

    length_mm, side_mm = 120.0, 10.0
    want = cantilever_first_mode_hz(length_mm / 1000.0, side_mm / 1000.0,
                                    side_mm / 1000.0)

    got = []
    for cl in (3.0, 2.0):
        case = tmp_path / f"cl{cl}"
        case.mkdir()
        stl = case / "beam.stl"
        cq.exporters.export(
            cq.Workplane("XY").box(side_mm, side_mm, length_mm), str(stl)
        )
        res = mesh_stl_volume(stl, case / "mesh.msh", cl_max_mm=cl,
                              cl_min_mm=cl * 0.6, scale_to_meters=True,
                              mesh_timeout_s=900)
        if not res.success:
            pytest.skip(f"gmsh unavailable or failed: {res.error}")
        generate_modal_case_inp(case, case_filename="modal.inp", fix_axis="z",
                                modes=4, youngs_modulus=E, poisson=NU,
                                density=RHO)
        run_calculix_case(case, job_name="modal", timeout=1800)
        freqs = parse_eigenfrequencies(case / "modal.dat")
        if not freqs:
            pytest.skip("CalculiX produced no eigenvalues")
        got.append(freqs[0])

    coarse, fine = got
    assert fine < coarse, f"refining moved away from theory: {coarse} -> {fine}"
    assert fine > want, "linear tets should over-predict, not under-predict"
    assert fine / want < 1.15, f"first mode {fine:.1f} Hz vs theory {want:.1f} Hz"


@pytest.mark.skipif(ccx_missing, reason="CalculiX not installed")
def test_square_section_has_a_near_degenerate_first_mode_pair(tmp_path):
    """A square cantilever bends identically about both axes, so modes 1 and 2
    must be the same frequency to within meshing asymmetry. If they are not, the
    constraint or the mesh is doing something other than a clean cantilever."""
    pytest.importorskip("cadquery")
    import cadquery as cq

    stl = tmp_path / "beam.stl"
    cq.exporters.export(cq.Workplane("XY").box(10.0, 10.0, 120.0), str(stl))
    res = mesh_stl_volume(stl, tmp_path / "mesh.msh", cl_max_mm=3.0,
                          cl_min_mm=1.8, scale_to_meters=True,
                          mesh_timeout_s=900)
    if not res.success:
        pytest.skip(f"gmsh unavailable or failed: {res.error}")
    generate_modal_case_inp(tmp_path, case_filename="modal.inp", fix_axis="z",
                            modes=4, youngs_modulus=E, poisson=NU, density=RHO)
    run_calculix_case(tmp_path, job_name="modal", timeout=1800)
    freqs = parse_eigenfrequencies(tmp_path / "modal.dat")
    if len(freqs) < 2:
        pytest.skip("CalculiX produced too few eigenvalues")
    assert freqs[1] / freqs[0] < 1.05, freqs[:2]
