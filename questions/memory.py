import joblib

from questions import config

memory = joblib.Memory(location=config.cache_dir(), verbose=0)
