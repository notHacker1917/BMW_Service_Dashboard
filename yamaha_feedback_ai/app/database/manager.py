"""Database initialization and management."""
import sqlite3
from pathlib import Path
from app.utils.logger import logger
from app.utils.config import DATABASE_PATH

class DatabaseManager:
    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = db_path
        self.init_database()

    def get_connection(self):
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_database(self):
        """Initialize database schema."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Raw feedback table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS raw_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feedback_id TEXT UNIQUE NOT NULL,
                    timestamp TEXT NOT NULL,
                    language TEXT NOT NULL,
                    vehicle_model TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    customer_feedback TEXT NOT NULL,
                    country TEXT NOT NULL,
                    mileage INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Cleaned feedback table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cleaned_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feedback_id TEXT UNIQUE NOT NULL,
                    cleaned_text TEXT NOT NULL,
                    language TEXT NOT NULL,
                    vehicle_model TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    country TEXT NOT NULL,
                    mileage INTEGER,
                    cleaned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (feedback_id) REFERENCES raw_feedback(feedback_id)
                )
            """)

            # Structured entities table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS structured_entities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feedback_id TEXT UNIQUE NOT NULL,
                    component TEXT,
                    failure_mode TEXT,
                    symptom TEXT,
                    severity TEXT,
                    driving_condition TEXT,
                    confidence REAL,
                    extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (feedback_id) REFERENCES cleaned_feedback(feedback_id)
                )
            """)

            # Embeddings table (stores metadata only, embeddings in numpy)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feedback_id TEXT UNIQUE NOT NULL,
                    embedding_index INTEGER NOT NULL,
                    dimension INTEGER NOT NULL,
                    embedded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (feedback_id) REFERENCES structured_entities(feedback_id)
                )
            """)

            # Clusters table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS clusters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feedback_id TEXT UNIQUE NOT NULL,
                    cluster_id INTEGER,
                    cluster_confidence REAL,
                    umap_x REAL,
                    umap_y REAL,
                    clustered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (feedback_id) REFERENCES embeddings(feedback_id)
                )
            """)

            # Cluster labels table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cluster_labels (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cluster_id INTEGER UNIQUE NOT NULL,
                    label TEXT NOT NULL,
                    root_component TEXT,
                    recurring_symptom TEXT,
                    failure_frequency INTEGER,
                    representative_complaint TEXT,
                    confidence REAL,
                    labeled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.commit()
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()

    def insert_raw_feedback(self, records: list):
        """Insert raw feedback records."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.executemany("""
                INSERT OR IGNORE INTO raw_feedback 
                (feedback_id, timestamp, language, vehicle_model, domain, customer_feedback, country, mileage)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, records)
            conn.commit()
            logger.info(f"Inserted {cursor.rowcount} raw feedback records")
            return cursor.rowcount
        except Exception as e:
            logger.error(f"Error inserting raw feedback: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()

    def get_table_count(self, table_name: str) -> int:
        """Get row count for a table."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) as count FROM {table_name}")
        count = cursor.fetchone()["count"]
        conn.close()
        return count

    def execute_query(self, query: str, params: tuple = ()):
        """Execute a query and return results."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()
        return results

    def close(self):
        """Close database connection."""
        pass
