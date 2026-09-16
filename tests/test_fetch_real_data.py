import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

from fetch_real_data import convert_detections_df, ALERCE_CLASS_MAP


def test_convert_detections_df_basic():
    df = pd.DataFrame({
        "mjd": [59000.1, 59001.3, 59002.7],
        "fid": [1, 2, 1],          # 1=g, 2=r
        "magpsf": [19.5, 19.2, 18.8],
        "sigmapsf": [0.05, 0.06, 0.04],
    })
    dets = convert_detections_df(df)
    assert len(dets) == 3
    assert {d["band"] for d in dets} == {"g", "r"}
    assert all(d["flux"] > 0 for d in dets)
    assert all(d["detected"] is True for d in dets)
    # brighter (lower) magnitude should convert to higher flux
    assert dets[2]["flux"] > dets[0]["flux"]  # mag 18.8 > mag 19.5 in flux


def test_convert_detections_df_unknown_band_dropped():
    df = pd.DataFrame({
        "mjd": [59000.0, 59000.5],
        "fid": [1, 99],  # 99 is not in BAND_MAP
        "magpsf": [19.0, 19.0],
        "sigmapsf": [0.05, 0.05],
    })
    dets = convert_detections_df(df)
    assert len(dets) == 1  # the fid=99 row is dropped, not crashed on


def test_convert_detections_df_empty():
    assert convert_detections_df(None) == []
    assert convert_detections_df(pd.DataFrame()) == []


def test_class_map_covers_all_project_labels():
    # every label this project trains on (Ia/Ibc/II) must be reachable
    # from some ALeRCE class name
    assert set(ALERCE_CLASS_MAP.values()) == {"Ia", "Ibc", "II"}
