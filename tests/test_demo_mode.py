"""RNF3 · Modo demostración (KOREX_DEMO=1): ninguna evaluación se guarda ni se muestra."""
from test_pipeline import PATIENT, pipe  # noqa: F401  (fixture compartido)


def test_demo_mode_does_not_write_or_read_audit(pipe, monkeypatch):  # noqa: F811
    before = len(pipe.read_audit())
    monkeypatch.setenv("KOREX_DEMO", "1")
    assert pipe.demo_mode()
    res = pipe.predict(PATIENT, "A")
    assert pipe.log_audit(PATIENT, res) == ""
    assert pipe.read_audit() == []
    monkeypatch.delenv("KOREX_DEMO")
    assert not pipe.demo_mode()
    assert len(pipe.read_audit()) == before
