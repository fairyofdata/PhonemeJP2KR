"""Score-reference bands and previous-attempt lookup."""

import json
import os
import tempfile

from src.reference import load_reference, score_band

REF = {"p10": 68, "median": 81, "n_faithful": 483}


def test_reference_reads_committed_exp6_results():
    ref = load_reference()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "experiments", "results", "exp6_l2_validation.json"),
              encoding="utf-8") as f:
        floor = json.load(f)["asr_noise_floor"]
    assert ref["p10"] == floor["p10"] and ref["median"] == floor["median"]


def test_reference_falls_back_when_results_missing():
    assert load_reference("does/not/exist.json") == REF


def test_band_boundaries():
    assert score_band(100, REF)["band"] == "typical_faithful"
    assert score_band(81, REF)["band"] == "typical_faithful"
    assert score_band(80, REF)["band"] == "indeterminate"
    assert score_band(68, REF)["band"] == "indeterminate"
    assert score_band(67, REF)["band"] == "rare_for_faithful"
    assert score_band(0, REF)["band"] == "rare_for_faithful"


def test_band_ranges_tile_0_to_100():
    covered = set()
    for s in range(101):
        b = score_band(s, REF)
        assert b["low"] <= s <= b["high"]
        covered.update(range(b["low"], b["high"] + 1))
    assert covered == set(range(101))


def test_previous_score_is_same_sentence_most_recent(monkeypatch):
    from src import database as db

    monkeypatch.setattr(db, "DB_PATH", os.path.join(tempfile.mkdtemp(), "h.db"))
    db.init_db()
    assert db.get_previous_score("감사합니다") is None
    db.save_record("감사합니다", "감사함니다", 70, "")
    db.save_record("안녕하세요", "안녕하세요", 95, "")
    db.save_record("감사합니다", "감사합니다", 76, "")
    assert db.get_previous_score("감사합니다") == 76
    assert db.get_previous_score("안녕하세요") == 95
