import sqlite3
from pathlib import Path

DB_PATH = Path("audio_event_spotter.db")  # Database filename


def get_connection():
    """Create or connect to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # enables dict-like row access
    return conn


def init_db():
    """Initialize database tables."""
    conn = get_connection()
    cur = conn.cursor()

    # Table: Audio Files
    cur.execute("""
        CREATE TABLE IF NOT EXISTS audio_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT NOT NULL,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # Table: Detected Events
    cur.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            audio_id INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            confidence REAL,
            timestamp_start REAL,
            timestamp_end REAL,
            FOREIGN KEY(audio_id) REFERENCES audio_files(id)
        );
    """)

    conn.commit()
    conn.close()
    print("Database initialized!")


if __name__ == "__main__":
    init_db()
