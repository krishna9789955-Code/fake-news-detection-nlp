"""
Database access layer using SQLite.
Stores prediction history locally for demonstration, logging, and dashboard statistics.
"""
import sqlite3
import os
from datetime import datetime
from config import DATABASE_PATH, DATABASE_DIR


def get_db_connection(db_path=DATABASE_PATH):
    """
    Creates and returns a thread-safe connection to the SQLite database.
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path=DATABASE_PATH):
    """
    Initializes the database schema if it doesn't already exist.
    """
    conn = get_db_connection(db_path)
    try:
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    title TEXT,
                    snippet TEXT,
                    input_mode TEXT NOT NULL,
                    prediction TEXT NOT NULL,
                    predicted_class INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    real_prob REAL NOT NULL,
                    fake_prob REAL NOT NULL,
                    top_indicator TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_class ON predictions(predicted_class)")
    finally:
        conn.close()


def record_prediction(
    title: str,
    snippet: str,
    input_mode: str,
    prediction: str,
    predicted_class: int,
    confidence: float,
    real_prob: float,
    fake_prob: float,
    top_indicator: str = "",
    db_path=DATABASE_PATH
) -> int:
    """
    Records a prediction entry in SQLite.
    Truncates snippet to prevent bloated storage.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    clean_title = (title or "").strip()[:180]
    clean_snippet = (snippet or "").strip()[:240]
    
    conn = get_db_connection(db_path)
    try:
        with conn:
            cur = conn.execute("""
                INSERT INTO predictions (
                    timestamp, title, snippet, input_mode,
                    prediction, predicted_class, confidence,
                    real_prob, fake_prob, top_indicator
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                now, clean_title, clean_snippet, input_mode,
                prediction, predicted_class, round(float(confidence), 2),
                round(float(real_prob), 2), round(float(fake_prob), 2),
                (top_indicator or "")[:60]
            ))
            return cur.lastrowid
    finally:
        conn.close()


def get_history(limit: int = 50, filter_class: str = None, search: str = None, db_path=DATABASE_PATH) -> list[dict]:
    """
    Retrieves recorded predictions with optional filtering by class and keyword search.
    """
    init_db(db_path)
    conn = get_db_connection(db_path)
    try:
        query = "SELECT * FROM predictions"
        params = []
        conditions = []
        
        if filter_class in ('0', '1'):
            conditions.append("predicted_class = ?")
            params.append(int(filter_class))
        elif filter_class == 'real':
            conditions.append("predicted_class = 0")
        elif filter_class == 'fake':
            conditions.append("predicted_class = 1")
            
        if search and search.strip():
            search_param = f"%{search.strip()}%"
            conditions.append("(title LIKE ? OR snippet LIKE ?)")
            params.extend([search_param, search_param])
            
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
            
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        
        cur = conn.cursor()
        cur.execute(query, params)
        rows = cur.fetchall()
        
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_history_stats(db_path=DATABASE_PATH) -> dict:
    """
    Computes summary stats of all predictions analyzed so far.
    """
    init_db(db_path)
    conn = get_db_connection(db_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) AS total, SUM(CASE WHEN predicted_class = 0 THEN 1 ELSE 0 END) AS real_cnt, SUM(CASE WHEN predicted_class = 1 THEN 1 ELSE 0 END) AS fake_cnt, AVG(confidence) AS avg_conf FROM predictions")
        row = cur.fetchone()
        
        total = row['total'] or 0
        real_cnt = row['real_cnt'] or 0
        fake_cnt = row['fake_cnt'] or 0
        avg_conf = round(row['avg_conf'] or 0.0, 1)
        
        return {
            'total_analyses': total,
            'real_predictions': real_cnt,
            'fake_predictions': fake_cnt,
            'avg_confidence': avg_conf
        }
    finally:
        conn.close()


def clear_history(db_path=DATABASE_PATH) -> bool:
    """
    Clears all recorded predictions.
    """
    conn = get_db_connection(db_path)
    try:
        with conn:
            conn.execute("DELETE FROM predictions")
        return True
    finally:
        conn.close()
