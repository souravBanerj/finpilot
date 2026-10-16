"""Central configuration. Everything comes from environment variables / .env."""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # dotenv is optional for the deterministic core / tests
    pass

ROOT = Path(__file__).resolve().parent.parent
MESH_BASE_URL = "https://api.meshapi.ai/v1"
MESH_API_KEY = os.getenv("MESH_API_KEY", "")
MESH_MODEL = os.getenv("MESH_MODEL", "openai/gpt-4o-mini")
CURRENCY = os.getenv("CURRENCY", "₹")
DB_PATH = os.getenv("FINPILOT_DB", str(ROOT / "data" / "finpilot.db"))
DIGEST_DIR = ROOT / "data" / "digests"

# Analysis thresholds (documented in README "design decisions")
MOM_INCREASE_PCT = 25.0      # flag a category if latest month is >=25% above trailing avg...
MOM_MIN_ABS = 1000.0         # ...and the absolute change is meaningful
SPIKE_SIGMA = 1.5            # spike must also exceed baseline by this many historical std-devs
ANOMALY_MAD_Z = 3.5          # robust z-score threshold for single-transaction outliers
DUPLICATE_WINDOW_DAYS = 3    # same merchant + same amount inside this window = duplicate
