import numpy as np
import pandas as pd
import tensorflow as tf
import yaml


# numpy
yaml.add_multi_representer(np.bool, yaml.representer.SafeRepresenter.represent_bool)
yaml.add_multi_representer(np.integer, yaml.representer.SafeRepresenter.represent_int)
yaml.add_multi_representer(np.floating, yaml.representer.SafeRepresenter.represent_float)
yaml.add_multi_representer(np.ndarray, yaml.representer.SafeRepresenter.represent_list)

# pandas
yaml.add_multi_representer(pd.Series, yaml.representer.SafeRepresenter.represent_dict)

# tensorflow
yaml.add_multi_representer(tf.Tensor, lambda dumper, data: dumper.represent(data.numpy()))
