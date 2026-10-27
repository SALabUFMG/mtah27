import numpy as np
import pytest

from mtah27.submission import empty_labels, n_frames, write_submission


def test_tamanho_oficial():
    assert n_frames() == 199_319
    assert empty_labels().shape == (199_319,)


def test_escreve_contrato(tmp_path):
    labels = empty_labels()
    labels[11185:11200] = "H"
    p = write_submission(labels, tmp_path / "t.csv")
    raw = p.read_bytes()
    assert raw.startswith(b"frame,label\n0,X\n")
    assert b"\r" not in raw
    assert raw.count(b"\n") == 199_320


def test_rejeita_label_invalido(tmp_path):
    labels = empty_labels()
    labels[0] = "Z"
    with pytest.raises(ValueError):
        write_submission(labels, tmp_path / "t.csv")


def test_rejeita_tamanho_errado(tmp_path):
    with pytest.raises(ValueError):
        write_submission(np.array(["X"] * 10), tmp_path / "t.csv")
