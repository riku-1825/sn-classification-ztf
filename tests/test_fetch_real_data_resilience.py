import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fetch_real_data import (
    retry_with_backoff, _atomic_write_json, _load_existing_dataset,
)


def test_retry_with_backoff_succeeds_after_transient_failures(monkeypatch):
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("simulated transient DNS failure")
        return "success"

    # Avoid actually sleeping during the test
    monkeypatch.setattr(time, "sleep", lambda _: None)

    result = retry_with_backoff(flaky, max_retries=5, base_delay=0.01)
    assert result == "success"
    assert calls["n"] == 3  # failed twice, succeeded on the 3rd attempt


def test_retry_with_backoff_gives_up_after_max_retries(monkeypatch):
    def always_fails():
        raise ConnectionError("permanent-looking failure")

    monkeypatch.setattr(time, "sleep", lambda _: None)

    try:
        retry_with_backoff(always_fails, max_retries=3, base_delay=0.01)
        assert False, "expected ConnectionError to propagate"
    except ConnectionError:
        pass  # expected


def test_retry_with_backoff_does_not_retry_non_transient_errors(monkeypatch):
    calls = {"n": 0}

    def bad_request():
        calls["n"] += 1
        raise ValueError("malformed request -- not a network issue")

    monkeypatch.setattr(time, "sleep", lambda _: None)

    try:
        retry_with_backoff(bad_request, max_retries=5, base_delay=0.01)
        assert False, "expected ValueError to propagate immediately"
    except ValueError:
        pass
    assert calls["n"] == 1  # NOT retried -- ValueError is not a transient network error


def test_atomic_write_and_load_roundtrip(tmp_path):
    out_path = str(tmp_path / "test_dataset.json")
    data = [{"object_id": "A", "label": "Ia", "detections": []}]
    _atomic_write_json(data, out_path)

    assert os.path.exists(out_path)
    assert not os.path.exists(out_path + ".tmp")  # temp file cleaned up (renamed)

    loaded = json.load(open(out_path))
    assert loaded == data


def test_load_existing_dataset_resume_tracking(tmp_path):
    out_path = str(tmp_path / "partial_dataset.json")
    data = [
        {"object_id": "A", "label": "Ia", "detections": []},
        {"object_id": "B", "label": "Ia", "detections": []},
        {"object_id": "C", "label": "Ibc", "detections": []},
    ]
    _atomic_write_json(data, out_path)

    loaded_dataset, done_by_label = _load_existing_dataset(out_path)
    assert len(loaded_dataset) == 3
    assert done_by_label["Ia"] == {"A", "B"}
    assert done_by_label["Ibc"] == {"C"}
    assert "II" not in done_by_label  # no II objects fetched yet


def test_load_existing_dataset_missing_file_returns_empty():
    dataset, done_by_label = _load_existing_dataset("/tmp/definitely_does_not_exist_12345.json")
    assert dataset == []
    assert done_by_label == {}


def test_load_existing_dataset_corrupt_file_starts_fresh(tmp_path):
    out_path = str(tmp_path / "corrupt.json")
    with open(out_path, "w") as f:
        f.write("{not valid json!!")

    dataset, done_by_label = _load_existing_dataset(out_path)
    assert dataset == []
    assert done_by_label == {}


def test_build_real_dataset_natural_mode_allocates_proportionally(tmp_path, monkeypatch):
    """Natural balance mode should allocate --total_budget across classes
    in proportion to their relative candidate counts, not equally.
    """
    import fetch_real_data as frd

    candidates = {
        "SNIa": [f"IA_{i}" for i in range(20)],
        "SNIbc": [f"IBC_{i}" for i in range(10)],
        "SNII": [f"II_{i}" for i in range(15)],
    }

    def fake_fetch_light_curve(oid):
        return [{"mjd": 59000.0, "band": "g", "flux": 1.0, "flux_err": 0.1, "detected": True}] * 3

    def fake_query_objects(alerce_class, page_size, max_pages=5):
        return candidates[alerce_class]

    monkeypatch.setattr(frd, "fetch_light_curve", fake_fetch_light_curve)
    monkeypatch.setattr(frd, "query_objects", fake_query_objects)

    out_path = str(tmp_path / "natural_dataset.json")
    dataset = frd.build_real_dataset(
        n_per_class=10, min_detections=1, out_path=out_path,
        save_every=5, balance_mode="natural",
    )

    from collections import Counter
    counts = Counter(d["label"] for d in dataset)
    # total_budget defaults to n_per_class * 3 = 30, candidates ratio 20:10:15
    # (total 45) -> expected proportional split 13:7:10
    assert counts["Ia"] == 13
    assert counts["Ibc"] == 7
    assert counts["II"] == 10


def test_build_real_dataset_equal_mode_still_balanced(tmp_path, monkeypatch):
    """balance_mode='equal' (the default) should be unaffected by the
    natural-mode refactor -- still exactly n_per_class per class.
    """
    import fetch_real_data as frd

    candidates = {
        "SNIa": [f"IA_{i}" for i in range(20)],
        "SNIbc": [f"IBC_{i}" for i in range(10)],
        "SNII": [f"II_{i}" for i in range(15)],
    }

    def fake_fetch_light_curve(oid):
        return [{"mjd": 59000.0, "band": "g", "flux": 1.0, "flux_err": 0.1, "detected": True}] * 3

    def fake_query_objects(alerce_class, page_size, max_pages=5):
        return candidates[alerce_class]

    monkeypatch.setattr(frd, "fetch_light_curve", fake_fetch_light_curve)
    monkeypatch.setattr(frd, "query_objects", fake_query_objects)

    out_path = str(tmp_path / "equal_dataset.json")
    dataset = frd.build_real_dataset(
        n_per_class=8, min_detections=1, out_path=out_path,
        save_every=5, balance_mode="equal",
    )

    from collections import Counter
    counts = Counter(d["label"] for d in dataset)
    assert counts["Ia"] == 8
    assert counts["Ibc"] == 8
    assert counts["II"] == 8
