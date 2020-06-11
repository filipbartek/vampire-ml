import appdirs
import joblib

from proving import config

memory = joblib.Memory(location=appdirs.user_cache_dir(config.program_name(), version=config.program_version()),
                       verbose=0)
