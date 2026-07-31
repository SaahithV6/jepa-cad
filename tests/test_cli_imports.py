from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

from cadflow.cli import main


class _SweepResult:
    ok = True
    sweep_cases = 0
    run_ok = 0
    verified = 0
    promoted = 0

    def to_dict(self) -> dict[str, object]:
        return {"ok": self.ok, "sweep_cases": self.sweep_cases}


def test_cli_imports_and_corpus_sweep_dispatch_without_torch(tmp_path: Path, monkeypatch) -> None:
    called = {}

    def fake_backend(*, prefer_real: bool):
        called["backend"] = prefer_real
        return SimpleNamespace(name="mock")

    def fake_sweep(raw_dirs, out_dir, **kwargs):
        called["raw_dirs"] = list(raw_dirs)
        called["out_dir"] = str(out_dir)
        called["kwargs"] = kwargs
        return _SweepResult()

    monkeypatch.setattr("cadflow.cli.get_backend", fake_backend)
    monkeypatch.setattr("cadflow.cli.run_parametric_corpus_sweep", fake_sweep)

    code = main(
        [
            "corpus-sweep",
            "--raw-dir",
            str(tmp_path / "raw"),
            "--out-dir",
            str(tmp_path / "out"),
            "--max-sources",
            "0",
            "--mock-cad",
        ]
    )

    assert code == 0
    assert called["backend"] is False
    assert called["raw_dirs"] == [str(tmp_path / "raw")]
    assert called["out_dir"] == str(tmp_path / "out")
    assert called["kwargs"]["max_sources"] == 0
