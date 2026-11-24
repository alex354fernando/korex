"""Smoke + regresión del pipeline de inferencia (lo que usa el dashboard)."""
import copy
import json
import os
import tempfile

import numpy as np
import pytest

TMP = tempfile.mkdtemp(prefix="korex_test_")
os.environ["KOREX_ARTIFACTS"] = TMP

from korex import load_config  # noqa: E402
from korex.kdigo import clean_dataframe, make_synthetic_uci  # noqa: E402
from korex.labeling import label_dataset  # noqa: E402
from korex.pipeline import InferencePipeline, KorexModel, default_params, payload_hash  # noqa: E402

PATIENT = {"age": 67, "sex": 1, "creatinine": 1.6, "hemo": 11.2, "bu": 60, "htn": "yes", "al": 2}


@pytest.fixture(scope="module")
def pipe():
    cfg = copy.deepcopy(load_config())
    cfg["preprocessing"]["missforest"].update(n_estimators=10, max_iter=2)
    cfg["explain"].update(shap_max_evals=200, lime_num_samples=300, background_size=20)
    data = label_dataset(clean_dataframe(make_synthetic_uci(600, 7), cfg), cfg)
    params = default_params(cfg, quick=True)
    for m in ("xgb", "lgbm", "cat", "et"):
        params[m] = {k: (20 if k in ("n_estimators", "iterations") else v) for k, v in params[m].items()}
    model = KorexModel(cfg, "A", params, 0, stack_folds=2).fit(data.X, data.y)
    p = InferencePipeline({"A": model}, {"A": model.transform(data.X.iloc[:20])},
                          {"version": "test", "trained_at": "2026-01-01", "evaluation": {},
                           "dataset": {"path": "synthetic", "n": len(data.y)}}, cfg)
    p.save(TMP)
    return InferencePipeline.load(TMP, cfg)


def test_single_model_features(pipe):
    feats = pipe.features("A")
    assert "creatinine" in feats and "egfr_ckdepi2021" not in feats and list(pipe.models) == ["A"]


def test_predict_five_stages(pipe):
    r = pipe.predict(PATIENT)
    assert r["risk_class"] in range(5) and len(r["probabilities"]) == 5
    assert abs(sum(r["probabilities"]) - 1) < 1e-3
    assert r["kdigo_rule"]["G"] == "G3" and r["kdigo_rule"]["sex_assumed"] == "registrado"
    assert len(r["payload_hash"]) == 64


def test_missing_age_is_rejected(pipe):
    with pytest.raises(ValueError):
        pipe.predict({"sex": 0, "creatinine": 1.0})


def test_determinism_and_id_invariance(pipe):
    a = pipe.predict({**PATIENT, "patient_id": "DNI-1"})
    b = pipe.predict({**PATIENT, "patient_id": "DNI-2"})
    assert a["probabilities"] == b["probabilities"]


def test_explain_report_pdf(pipe):
    r = pipe.predict(PATIENT)
    ex = pipe.explain(PATIENT)
    assert ex["factors"] and 0 <= ex["convergence"]["icsl"] <= 1
    assert "Hipertensión (Sí)" in ex["reading"] or "Hipertensión" not in ex["reading"]
    rep = pipe.build_report(r, ex, PATIENT)
    assert any(row["variable"] == "Sexo" and row["valor"] == "Mujer" for row in rep["datos_paciente"])
    pdf = pipe.to_pdf(rep, r["color"])
    assert pdf[:4] == b"%PDF" and len(pdf) > 1000
    assert "Datos del paciente" in pipe.render_report_html(rep, r["color"])
    assert pipe.to_fhir(r, rep)["resourceType"] == "DiagnosticReport"


def test_audit_chain_without_identifiers(pipe):
    for _ in range(3):
        pipe.log_audit({**PATIENT, "patient_id": "DNI-9"}, pipe.predict(PATIENT))
    lines = [json.loads(x) for x in open(os.path.join(TMP, "auditoria.jsonl"), encoding="utf-8")]
    assert "DNI-9" not in json.dumps(lines)
    for prev, cur in zip(lines, lines[1:]):
        assert cur["prev_hash"] == prev["record_hash"]
    body = {k: v for k, v in lines[-1].items() if k != "record_hash"}
    assert payload_hash(body) == lines[-1]["record_hash"]


def test_batch(pipe):
    df = make_synthetic_uci(30, 3)
    out = pipe.predict_batch(df)
    assert set(out["estadio"]) <= {"G1", "G2", "G3", "G4", "G5"} and len(out) <= 30
    assert np.isfinite(out[["p0", "p1", "p2", "p3", "p4"]].to_numpy()).all()
