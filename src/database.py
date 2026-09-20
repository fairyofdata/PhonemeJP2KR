"""SQLite persistence for practice history.

A record keeps both the log fields (sentence, ASR output, score, feedback)
and the full analysis payload, so a past attempt can be reopened in the
result view. The recording itself is kept as a file in CLIPS_DIR for the
most recent KEEP_CLIPS records; older clips are pruned, and their record
reopens without the waveform player.
"""

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime

from .config import CLIPS_DIR, DB_PATH

KEEP_CLIPS = 50


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                intended TEXT,
                actual TEXT,
                score INTEGER,
                feedback TEXT
            )
        """)
        # migration: per-attempt L1 error tags feed the weak-point profile
        cols = {r[1] for r in conn.execute("PRAGMA table_info(feedback_history)")}
        if "error_tags" not in cols:
            conn.execute("ALTER TABLE feedback_history ADD COLUMN error_tags TEXT")
        # migration: full analysis payload, so a record can be reopened
        if "analysis" not in cols:
            conn.execute("ALTER TABLE feedback_history ADD COLUMN analysis TEXT")


def save_record(intended: str, actual: str, score: int, feedback: str,
                error_tags: list = None, analysis: dict = None) -> int:
    """Insert one attempt and return its id."""
    tag_names = [t["tag"] for t in (error_tags or [])]
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO feedback_history"
            " (timestamp, intended, actual, score, feedback, error_tags, analysis)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
             intended, actual, score, feedback,
             json.dumps(tag_names, ensure_ascii=False),
             json.dumps(analysis, ensure_ascii=False) if analysis else None),
        )
        return cur.lastrowid


def get_record(record_id: int):
    """One record as a dict, with `analysis` decoded (None when absent)."""
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM feedback_history WHERE id = ?", (record_id,)
        ).fetchone()
    if row is None:
        return None
    record = dict(row)
    raw = record.get("analysis")
    try:
        record["analysis"] = json.loads(raw) if raw else None
    except ValueError:
        record["analysis"] = None
    return record


# --- recordings on disk ------------------------------------------------------

def clip_path(record_id: int, suffix: str = ".wav") -> str:
    return os.path.join(CLIPS_DIR, f"{record_id}{suffix}")


def save_clip(record_id: int, audio_bytes: bytes, suffix: str = ".wav") -> str:
    """Store the recording for a record and prune clips of older records."""
    os.makedirs(CLIPS_DIR, exist_ok=True)
    path = clip_path(record_id, suffix)
    with open(path, "wb") as f:
        f.write(audio_bytes)
    prune_clips()
    return path


def find_clip(record_id: int):
    """Path of this record's stored recording, or None if it was pruned."""
    if not os.path.isdir(CLIPS_DIR):
        return None
    for name in os.listdir(CLIPS_DIR):
        stem, _, _ = name.partition(".")
        if stem == str(record_id):
            return os.path.join(CLIPS_DIR, name)
    return None


def delete_clip(record_id: int):
    path = find_clip(record_id)
    if path:
        os.remove(path)


def prune_clips(keep: int = KEEP_CLIPS):
    """Keep recordings for the `keep` most recent records only."""
    if not os.path.isdir(CLIPS_DIR):
        return
    with _connect() as conn:
        recent = {str(r[0]) for r in conn.execute(
            "SELECT id FROM feedback_history ORDER BY id DESC LIMIT ?", (keep,))}
    for name in os.listdir(CLIPS_DIR):
        if name.partition(".")[0] not in recent:
            os.remove(os.path.join(CLIPS_DIR, name))


def get_all_records():
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM feedback_history ORDER BY id DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def delete_record(record_id: int):
    delete_clip(record_id)
    with _connect() as conn:
        conn.execute("DELETE FROM feedback_history WHERE id = ?", (record_id,))


def get_previous_score(intended: str):
    """Score of the most recent earlier attempt at the same sentence, or None.

    Scores are only comparable within one sentence (the ASR noise floor
    varies by sentence), so the match is on the exact target text.
    """
    with _connect() as conn:
        row = conn.execute(
            "SELECT score FROM feedback_history WHERE intended = ?"
            " ORDER BY id DESC LIMIT 1",
            (intended,),
        ).fetchone()
    return row[0] if row else None


def get_weak_points(recent: int = 30):
    """Count L1 error tags over the most recent attempts → [(tag, count)].

    Rows saved before the error_tags migration are skipped.
    """
    with _connect() as conn:
        rows = conn.execute(
            "SELECT error_tags FROM feedback_history"
            " WHERE error_tags IS NOT NULL ORDER BY id DESC LIMIT ?",
            (recent,),
        ).fetchall()
    counts = {}
    for (raw,) in rows:
        for tag in json.loads(raw):
            counts[tag] = counts.get(tag, 0) + 1
    return sorted(counts.items(), key=lambda kv: -kv[1])
