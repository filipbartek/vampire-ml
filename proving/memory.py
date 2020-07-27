import appdirs
import joblib

from proving import config

location = appdirs.user_cache_dir(config.program_name(), version=config.program_version())
memory = joblib.Memory(location=location, verbose=0)
