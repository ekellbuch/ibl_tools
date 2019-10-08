import numpy as np


def quantile_scaling(x, min_per=5, max_per=95):
    """
    Scale data using max and min quantiles
    """
    # see quantile_transform sklearn
    x_min = np.nanpercentile(x, min_per)
    x_max = np.nanpercentile(x, max_per)
    xrmp = (x - x_min) / (x_max - x_min)
    return xrmp, x_min, x_max
