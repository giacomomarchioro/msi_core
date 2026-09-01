from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile

from msi_core.cli import main


def _write_gray(path: Path, value: int = 1) -> None:
    tifffile.imwrite(path, np.full((8, 9), value, dtype=np.uint16))


def _write_rgb(path: Path) -> None:
    data = np.zeros((8, 9, 3), dtype=np.uint16)
    data[..., 0] = 100
    data[..., 1] = 200
    data[..., 2] = 300
    tifffile.imwrite(path, data, photometric="rgb")


def test_cli_bands_show(tmp_path, capsys):
    _write_gray(tmp_path / "365 UV.tif")
    _write_gray(tmp_path / "475 Blue.tif")
    _write_gray(tmp_path / "630 Red.tif")

    assert main(["bands", "show", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert "365 UV.tif" in output
    assert "365 nm" in output


def test_cli_bands_show_json(tmp_path, capsys):
    _write_gray(tmp_path / "365 UV.tif")
    assert main(["bands", "show", str(tmp_path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["sources"][0]["wavelength_nm"] == 365.0


def test_cli_bands_init_and_validate(tmp_path, capsys):
    _write_gray(tmp_path / "365 UV.tif")
    _write_gray(tmp_path / "475 Blue.tif")
    _write_gray(tmp_path / "535 Green.tif")
    _write_gray(tmp_path / "630 Red.tif")
    _write_rgb(tmp_path / "VIS.tif")

    assert main(["bands", "init", str(tmp_path), "--yes"]) == 0
    manifest = tmp_path / "bands.json"
    assert manifest.is_file()

    assert main(["bands", "validate", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert "Manifest written" in output
    assert "OK:" in output

    payload = json.loads(manifest.read_text())
    assert payload["schema_version"] == "2.0"
    assert any(rep["id"] == "visible_rgb" for rep in payload["representations"])


def test_cli_init_does_not_overwrite_without_force(tmp_path, capsys):
    _write_gray(tmp_path / "365 UV.tif")
    (tmp_path / "bands.json").write_text("{}")

    assert main(["bands", "init", str(tmp_path), "--yes"]) == 2
    assert "--force" in capsys.readouterr().err
