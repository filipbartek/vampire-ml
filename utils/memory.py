import appdirs
import joblib

memory = joblib.Memory(location=appdirs.user_cache_dir('vampire_ml'), verbose=0)
memory.recompute = False
