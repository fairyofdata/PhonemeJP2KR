"""SQLite persistence for practice history."""

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


def save_record(intended: str, actual: str, score: int, feedback: str):
    with _connect() as conn:
        conn.execute(
            "INSERT INTO feedback_history (timestamp, intended, actual, score, feedback)"
            " VALUES (?, ?, ?, ?, ?)",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
             intended, actual, score, feedback),
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
