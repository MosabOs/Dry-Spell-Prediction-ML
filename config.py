
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent

# Main directories
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Dataset file extensions
DATA_EXTENSION = ".csv"
