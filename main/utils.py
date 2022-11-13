import inspect
from pathlib import Path


ROOT = Path(__file__).parents[1]
LOG_DIR = Path(ROOT) / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path(ROOT) / 'data'
DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_func_kwargs(func, kwargs):
    return {
        arg: kwargs[arg] for arg in
        set(inspect.getfullargspec(func).args) & set(kwargs)}