from pathlib import Path


ROOT = Path(__file__).parents[1]
LOG_DIR = Path(ROOT) / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path(ROOT) / 'data'
DATA_DIR.mkdir(parents=True, exist_ok=True)
