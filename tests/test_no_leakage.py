"""C2 · El transformador se ajusta solo con entrenamiento y transforma fila a fila."""
import copy

import numpy as np
import pytest

from korex import load_config
from korex.kdigo import clean_dataframe, make_synthetic_uci
from korex.labeling import label_dataset, scenario_features
from korex.preprocessing import Preprocessor

CFG = copy.deepcopy(load_config())
CFG["preprocessing"]["missforest"].update(n_estimators=15, max_iter=3)


@pytest.fixture(scope="module")
def split():
    data = label_dataset(clean_dataframe(make_synthetic_uci(500, 3), CFG), CFG)
    X = data.X[scenario_features(data.X.columns, CFG, "B")]
    return X.iloc[:320].reset_index(drop=True), X.iloc[320:].reset_index(drop=True)


def _params(pre):
    return (pre.median_.copy(), pre.iqr_.copy(), list(pre.log_cols_), pre.imputer_.init_.copy())


def test_parameters_do_not_change_with_test_data(split):
    train, test = split
    pre = Preprocessor(CFG, 0).fit(train)
    before = _params(pre)
    corrupted = test.copy()
    corrupted[corrupted.select_dtypes("number").columns] *= 1000          # datos de prueba absurdos
    pre.transform(corrupted)
    after = _params(pre)
    assert before[0] == after[0] and before[1] == after[1] and before[2] == after[2]
    assert np.array_equal(before[3], after[3])


def test_transform_is_row_independent(split):
    """La transformación de un paciente no depende de los demás pacientes de prueba."""
    train, test = split
    pre = Preprocessor(CFG, 0).fit(train)
    full = pre.transform(test).to_numpy()
    single = np.vstack([pre.transform(test.iloc[[i]]).to_numpy() for i in range(5)])
    assert np.allclose(full[:5], single)


def test_output_complete_and_typed(split):
    train, test = split
    pre = Preprocessor(CFG, 0).fit(train)
    Xt = pre.transform(test)
    assert not Xt.isna().any().any()
    for col, t in pre.out_types_.items():
        if t in ("bin", "nom"):
            assert set(np.unique(Xt[col])) <= {0.0, 1.0}
    for group in pre.onehot_groups_.values():
        assert np.allclose(Xt[group].sum(axis=1), 1.0)


def test_unseen_category_and_missing_columns(split):
    train, test = split
    pre = Preprocessor(CFG, 0).fit(train)
    row = test.iloc[[0]].copy()
    row["htn"] = "maybe"                                 # categoría no vista → se imputa
    row = row.drop(columns=["hemo"])                     # columna ausente → se imputa
    Xt = pre.transform(row)
    assert not Xt.isna().any().any()


def test_leakage_columns_never_features():
    data = label_dataset(clean_dataframe(make_synthetic_uci(200, 4), CFG), CFG)
    leak = {c.lower() for c in CFG["data"]["leakage_columns"]}
    assert not {c.lower() for c in data.X.columns} & leak
