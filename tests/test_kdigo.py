"""C1 · CKD-EPI 2021, estadios G1–G5, supuesto de sexo, lectura ARFF y limpieza."""
import numpy as np
import pandas as pd
import pytest

from korex import load_config
from korex.kdigo import ckd_epi_2021, clean_dataframe, egfr_by_policy, g_stage, make_synthetic_uci, read_arff
from korex.labeling import V_DEF, circularity_index, label_dataset, scenario_features

CFG = load_config()


@pytest.mark.parametrize("scr,age,female,expected", [
    (1.0, 50, 0, 91.7),     # hombre 50 a, 1,0 mg/dL ≈ 92 (calculadora NKF)
    (1.2, 60, 1, 51.9),     # mujer 60 a, 1,2 mg/dL ≈ 52
    (0.7, 40, 1, 112.1),    # en κ: solo factor de edad y sexo
])
def test_ckd_epi_2021_reference_values(scr, age, female, expected):
    assert ckd_epi_2021([scr], [age], [female])[0] == pytest.approx(expected, abs=1.0)


def test_ckd_epi_formula_exact():
    assert ckd_epi_2021([0.9], [45], [0])[0] == pytest.approx(142 * 0.9938 ** 45, rel=1e-9)


@pytest.mark.parametrize("egfr,stage", [(120, 0), (90, 0), (89.9, 1), (60, 1), (59.9, 2), (30, 2),
                                        (29.9, 3), (15, 3), (14.9, 4), (3, 4)])
def test_g_stage_boundaries(egfr, stage):
    assert g_stage([egfr], CFG["kdigo"]["g_cuts"])[0] == stage


def test_g_stage_missing():
    assert np.isnan(g_stage([np.nan])[0])


def test_sex_unknown_worst_case_is_most_advanced_stage():
    scr, age = np.array([0.8, 1.2, 2.5, 6.0]), np.array([40, 55, 70, 30])
    worst = g_stage(egfr_by_policy(scr, age, None, "worst"))
    male = g_stage(egfr_by_policy(scr, age, None, "male"))
    female = g_stage(egfr_by_policy(scr, age, None, "female"))
    assert np.all(worst == np.maximum(male, female))
    known = egfr_by_policy(scr, age, np.array([0, 1, np.nan, 0]), "worst")   # sexo conocido se respeta
    assert known[0] == pytest.approx(ckd_epi_2021([0.8], [40], [0])[0])


def test_clinical_invariance_monotonic():
    """Más creatinina (misma edad) nunca da un estadio menos avanzado; más edad tampoco."""
    scr = np.linspace(0.3, 15, 200)
    for policy in ("worst", "male", "female"):
        st = g_stage(egfr_by_policy(scr, np.full_like(scr, 50), None, policy))
        assert np.all(np.diff(st) >= 0)
    ages = np.linspace(18, 95, 100)
    st = g_stage(egfr_by_policy(np.full_like(ages, 1.1), ages, None))
    assert np.all(np.diff(st) >= 0)


def test_read_arff_tolerates_extra_commas(tmp_path):
    f = tmp_path / "x.arff"
    f.write_text("@relation x\n@attribute 'age' numeric\n@attribute 'sc' numeric\n@attribute 'htn' {yes,no}\n"
                 "@attribute 'class' {ckd,notckd}\n@data\n50,1.2,yes,ckd\n60,?,\tno,ckd,\n45,0.9,,no,notckd\n")
    df = read_arff(f)
    assert len(df) == 3 and df.attrs["rejected_rows"] == 0
    assert df["htn"].tolist() == ["yes", "no", "no"]


def test_clean_dataframe_missing_codes_and_age():
    df = pd.DataFrame({"age": ["50", "?", "15", "70"], "sc": ["1.2", "2,5", "0.8", "\t?"],
                       "htn": [" yes", "no", "no", "YES"], "class": ["ckd", "ckd", "notckd", "ckd"]})
    out = clean_dataframe(df, CFG)
    assert len(out) == 2                                     # sin edad y <18 excluidos
    assert out["creatinine"].iloc[0] == pytest.approx(1.2) and np.isnan(out["creatinine"].iloc[1])
    assert out["htn"].tolist() == ["yes", "yes"]


def test_labeling_drops_leakage_and_defines_scenarios():
    data = label_dataset(clean_dataframe(make_synthetic_uci(300, 1), CFG), CFG)
    assert "class" not in data.X.columns
    vb = scenario_features(data.X.columns, CFG, "B")
    assert not set(vb) & set(V_DEF)
    va = scenario_features(data.X.columns, CFG, "A")
    assert "creatinine" in va and "egfr_ckdepi2021" not in va and len(va) > len(vb)
    assert set(np.unique(data.y)) <= set(range(5))
    assert data.info["sex_available"] is False


def test_circularity_index():
    assert circularity_index(0.9, 0.9, 0.25) == pytest.approx(0.0)
    assert circularity_index(0.9, 0.25, 0.25) == pytest.approx(1.0)
