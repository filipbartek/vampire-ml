import appdirs
import joblib

from .utils import makedirs_open, get_consistent, len_robust, fill_category_na, numpy_err_settings, truncate, \
    dict_to_name
from .progress_bar import ProgressBar

memory = joblib.Memory(location=appdirs.user_cache_dir('vampire_ml'), verbose=0)
