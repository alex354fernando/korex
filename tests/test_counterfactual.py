"""RF7 · Simulador y comparación: predicción vectorizada coherente y sin escritura en la auditoría."""
import numpy as np

from test_pipeline import PATIENT, pipe  # noqa: F401  (fixture compartido)


def test_batch_probabilities_match_single_prediction(pipe):  # noqa: F811
    records = [PATIENT, {**PATIENT, "creatinine": 3.5}, {**PATIENT, "age": 30, "creatinine": 0.8}]
    P = pipe.predict_proba_records(records, "A")
    assert P.shape == (3, 5)
    for row, rec in zip(P, records):
        assert np.allclose(np.round(row, 4), pipe.predict(rec, "A")["probabilities"], atol=1e-4)


def test_sweep_of_45_values_is_valid_distribution(pipe):  # noqa: F811
    grid = np.linspace(0.5, 10.0, 45)
    P = pipe.predict_proba_records([{**PATIENT, "creatinine": float(v)} for v in grid], "A")
    assert P.shape == (45, 5)
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-6) and (P >= 0).all()


def test_simulation_and_comparison_do_not_write_audit(pipe):  # noqa: F811
    before = len(pipe.read_audit())
    pipe.predict(PATIENT, "A")
    pipe.predict({**PATIENT, "hemo": 9.0}, "A")
    pipe.predict_proba_records([PATIENT] * 5, "A")
    assert len(pipe.read_audit()) == before
