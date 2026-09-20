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
