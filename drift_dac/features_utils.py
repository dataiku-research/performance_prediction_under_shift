import numpy as np


def is_integer_feature(x, f):
    values = np.unique(x[:, f])
    if values.dtype.char in np.typecodes['AllInteger']:
        return True
    else:
        return np.all([v.is_integer() for v in values])

