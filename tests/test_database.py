"""History persistence: stored analyses and the recordings kept for replay."""

import os

import pytest

from src import database as db


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "history.db"))
    monkeypatch.setattr(db, "CLIPS_DIR", str(tmp_path / "clips"))
    db.init_db()
    return db


def _save(store, sentence="감사합니다", score=80, analysis=None):
    return store.save_record(sentence, sentence, score, "fb", [],
                             analysis=analysis if analysis is not None else {"score": score})


def test_record_round_trips_the_analysis(store):
    payload = {"score": 76, "error_tags": [{"tag": "coda_deletion", "ref": "ㅂ", "hyp": ""}],
               "peaks": [0.1, 0.9], "target": "밥"}
    record_id = _save(store, analysis=payload)
    record = store.get_record(record_id)
    assert record["analysis"] == payload
    assert record["intended"] == "감사합니다"


def test_legacy_rows_without_analysis_reopen_as_none(store):
    store.save_record("밥", "바", 50, "fb", [])
    record = store.get_record(1)
    assert record["analysis"] is None


def test_get_record_missing_id(store):
    assert store.get_record(999) is None


def test_clip_is_saved_and_found(store):
    record_id = _save(store)
    store.save_clip(record_id, b"RIFF....WAVE", ".wav")
    found = store.find_clip(record_id)
    assert found and os.path.basename(found) == f"{record_id}.wav"


def test_clip_found_regardless_of_container(store):
    record_id = _save(store)
    store.save_clip(record_id, b"ID3 mp3 bytes", ".mp3")
    assert store.find_clip(record_id).endswith(f"{record_id}.mp3")


def test_pruning_keeps_only_recent_clips(store):
    ids = [_save(store) for _ in range(5)]
    for record_id in ids:
        store.save_clip(record_id, b"x", ".wav")
    store.prune_clips(keep=2)
    kept = {int(n.split(".")[0]) for n in os.listdir(store.CLIPS_DIR)}
    assert kept == set(ids[-2:])


def test_deleting_a_record_deletes_its_clip(store):
    record_id = _save(store)
    store.save_clip(record_id, b"x", ".wav")
    store.delete_record(record_id)
    assert store.find_clip(record_id) is None
    assert store.get_record(record_id) is None


def test_find_clip_without_clip_directory(store):
    assert store.find_clip(_save(store)) is None


def test_admin_takes_stay_out_of_the_learner_statistics(store):
    tags = [{"tag": "vowel_epenthesis", "ref": "", "hyp": "ㅜ"}]
    store.save_record("밥", "밥", 70, "fb", [{"tag": "coda_deletion", "ref": "ㅂ", "hyp": ""}],
                      analysis={"score": 70})
    store.save_record("밥", "바부", 40, "fb", tags, analysis={"score": 40},
                      source="admin")
    assert store.get_previous_score("밥") == 70            # not the admin take's 40
    assert dict(store.get_weak_points()) == {"coda_deletion": 1}
    sources = [r["source"] for r in store.get_all_records()]
    assert sources == ["admin", "recording"]               # still listed in the history


def test_rows_from_before_the_source_column_are_labelled(store, tmp_path, monkeypatch):
    import json
    import sqlite3

    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE feedback_history (id INTEGER PRIMARY KEY AUTOINCREMENT,"
                 " timestamp TEXT, intended TEXT, actual TEXT, score INTEGER, feedback TEXT,"
                 " error_tags TEXT, analysis TEXT)")
    conn.execute("INSERT INTO feedback_history (intended, analysis) VALUES (?, ?)",
                 ("밥", json.dumps({"admin_input": {"voice": "x"}, "previous_score": 76})))
    conn.execute("INSERT INTO feedback_history (intended, analysis) VALUES (?, ?)",
                 ("밥", json.dumps({"score": 60})))
    conn.commit()
    conn.close()
    monkeypatch.setattr(db, "DB_PATH", str(path))
    db.init_db()
    first, second = db.get_record(1), db.get_record(2)
    assert first["source"] == "admin" and first["analysis"]["previous_score"] is None
    assert second["source"] == "recording"
