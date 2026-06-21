import sqlite3
import os
from datetime import datetime

DB_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "history.db")

def init_db():
    os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS feedback_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            intended TEXT,
            actual TEXT,
            score INTEGER,
            feedback TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_record(intended, actual, score, feedback):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute('''
        INSERT INTO feedback_history (timestamp, intended, actual, score, feedback)
        VALUES (?, ?, ?, ?, ?)
    ''', (timestamp, intended, actual, score, feedback))
    conn.commit()
    conn.close()

def get_all_records():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('SELECT * FROM feedback_history ORDER BY id DESC')
    records = c.fetchall()
    conn.close()
    
    # dict list 형태로 변환하여 반환
    columns = ["id", "timestamp", "intended", "actual", "score", "feedback"]
    result = [dict(zip(columns, row)) for row in records]
    return result

def delete_record(record_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('DELETE FROM feedback_history WHERE id = ?', (record_id,))
    conn.commit()
    conn.close()
