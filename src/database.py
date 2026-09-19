"""SQLite persistence for practice history."""

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime

from .config import DB_PATH


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


def save_record(intended: str, actual: str, score: int, feedback: str,
                error_tags: list = None):
    tag_names = [t["tag"] for t in (error_tags or [])]
    with _connect() as conn:
        conn.execute(
            "INSERT INTO feedback_history"
            " (timestamp, intended, actual, score, feedback, error_tags)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
             intended, actual, score, feedback,
             json.dumps(tag_names, ensure_ascii=False)),
        )


def get_all_records():
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM feedback_history ORDER BY id DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def delete_record(record_id: int):
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
