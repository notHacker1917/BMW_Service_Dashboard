"""Configuration management."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DATA_DIR = DATA_DIR / "outputs"
DB_DIR = PROJECT_ROOT / "yamaha_feedback_ai" / "database"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUT_DATA_DIR, DB_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Database
DATABASE_PATH = os.getenv("DATABASE_PATH", str(DB_DIR / "feedback.db"))

# LLM
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = "gpt-4o-mini"
LLM_MAX_TOKENS = 500
LLM_TEMPERATURE = 0.3

# Embedding
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
EMBEDDING_BATCH_SIZE = 32
DEVICE = os.getenv("DEVICE", "cpu")

# Clustering
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_N_COMPONENTS = 15
UMAP_METRIC = "cosine"

HDBSCAN_MIN_CLUSTER_SIZE = 10
HDBSCAN_METRIC = "euclidean"
HDBSCAN_CLUSTER_SELECTION_METHOD = "eom"

# Refinement
CLUSTER_SIMILARITY_THRESHOLD = 0.90

# Data generation
SYNTHETIC_DATA_SIZE = 5000
FEEDBACK_ID_PREFIX = "YMH"

# Languages
SUPPORTED_LANGUAGES = ["en", "de", "da", "pl", "fr", "it", "es", "nl"]

# Vehicle models
VEHICLE_MODELS = [
    "Yamaha R15",
    "Yamaha MT-15",
    "Yamaha R3",
    "Yamaha MT-07",
    "Yamaha FZ-S",
    "Yamaha Tenere 700",
]

# Domains
DOMAINS = ["Interior", "PowerTrain", "Display & Infotainment"]

# Failure types
FAILURES = [
    "engine overheating",
    "startup issue",
    "display freezing",
    "ECU malfunction",
    "bluetooth disconnect",
    "clutch slipping",
    "brake vibration",
    "navigation lag",
    "fuel sensor issue",
    "ignition failure",
    "suspension noise",
    "coolant leak",
]

# Countries
COUNTRIES = ["Germany", "Denmark", "Poland", "France", "Italy", "Spain", "Netherlands"]

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
