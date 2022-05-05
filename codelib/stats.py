import numpy as np
from numpy import ndarray
from typing import Union


def weighted_percentile(x: np.array, p: Union[float, np.ndarray], probs=None, axis=0):

    """
    Function that calculates weighted percentiles

    Parameters
    ----------
    x:
        Array-like data for which to calculate percentiles.
    p:
        Percentile(s) to calculate.
    probs:
        Probabilities / weights
    axis
        Axis over which to calculate.

    Returns
    -------
    np.array
        Percentiles

    """
    x = np.asarray(x)
    ndim = x.ndim

    # make sure the probs are set
    if probs is None:
        if ndim == 1:
            probs = np.ones_like(x) / len(x)
        elif axis == 0:
            length = x.shape[0]
            probs = np.ones(length) / length
        elif axis == 1:
            length = x.shape[1]
            probs = np.ones(length) / length
        else:
            raise ValueError('probs cannot be set')

    if ndim == 1:

        # get sorted index
        index_sorted = np.argsort(x)

        # get sorted data (x)
        sorted_x = x[index_sorted]

        # sorted probs
        sorted_probs = probs[index_sorted]

        # get cumulated probs
        cum_sorted_probs = np.cumsum(sorted_probs)

        pn = (cum_sorted_probs - 0.5 * sorted_probs) / cum_sorted_probs[-1]

        return np.interp(p, pn, sorted_x, left=sorted_x[0], right=sorted_x[-1])

    else:

        return np.apply_along_axis(weighted_percentile, axis, x, p, probs)


def c_var(x: np.array, p: float, probs: np.array = None, axis: int = 0):
    """
    Conditional Value at Risk Function
    """
    var = weighted_percentile(x, p=p, probs=probs, axis=axis)
    bool_array = (x < var) * x if (axis == 0) else ((x.T < var) * x.T).T
    bool_array[bool_array == 0] = np.nan

    masked_data = np.ma.masked_array(bool_array, np.isnan(bool_array))

    if x.ndim == 1:
        result = np.ma.average(masked_data, axis=axis, weights=probs)
    else:
        result = np.ma.average(masked_data, axis=axis, weights=probs).filled(np.nan)

    return result


def weighted_mean(var, wts):
    """Calculates the weighted mean"""
    return np.average(var, weights=wts)


def weighted_variance(var, wts):
    """Calculates the weighted variance"""
    return np.average((var - weighted_mean(var, wts))**2, weights=wts)


def weighted_skew(var, wts):
    """Calculates the weighted skewness"""
    return (np.average((var - weighted_mean(var, wts))**3, weights=wts) /
            weighted_variance(var, wts)**(1.5))


def weighted_kurtosis(var, wts):
    """Calculates the weighted skewness"""
    return (np.average((var - weighted_mean(var, wts))**4, weights=wts) /
            weighted_variance(var, wts)**(2))