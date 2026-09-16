import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.preprocessing import (
    records_to_long_df, filter_objects, to_padded_sequences,
    PreprocessConfig, train_val_test_split_by_object,
)

FAKE_OBJECTS = [
    {
        "object_id": "A",
        "label": "Ia",
        "detections": [
            {"mjd": 100.0, "band": "g", "flux": 1.0, "flux_err": 0.1},
            {"mjd": 101.0, "band": "g", "flux": 2.0, "flux_err": 0.1},
            {"mjd": 100.5, "band": "r", "flux": 1.5, "flux_err": 0.1},
        ],
    },
    {
        "object_id": "B",
        "label": "II",
        "detections": [
            {"mjd": 200.0, "band": "g", "flux": 0.5, "flux_err": 0.1},
        ],
    },
]


def test_records_to_long_df():
    df = records_to_long_df(FAKE_OBJECTS)
    assert df.shape[0] == 4
    assert set(df["object_id"]) == {"A", "B"}


def test_filter_objects_drops_short():
    df = records_to_long_df(FAKE_OBJECTS)
    cfg = PreprocessConfig(min_detections=2)
    filtered = filter_objects(df, cfg)
    assert set(filtered["object_id"]) == {"A"}  # B has only 1 detection


def test_padded_sequences_shape():
    df = records_to_long_df(FAKE_OBJECTS)
    cfg = PreprocessConfig(min_detections=1, max_len=10)
    X, mask, y, oids = to_padded_sequences(df, cfg)
    assert X.shape == (2, 10, 3, 2)
    assert mask.shape == (2, 10, 3)
    assert len(y) == 2


def test_split_no_leakage():
    oids = [f"obj{i}" for i in range(20)]
    y = np.array(["Ia", "Ibc", "II"] * 6 + ["Ia", "Ia"])
    idx_train, idx_val, idx_test = train_val_test_split_by_object(oids, y)
    all_idx = set(idx_train) | set(idx_val) | set(idx_test)
    assert len(all_idx) == 20  # every object assigned exactly once
    assert set(idx_train) & set(idx_val) == set()
    assert set(idx_train) & set(idx_test) == set()
