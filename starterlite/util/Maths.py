import numpy as np

def central_difference(x, y):
    """
    Compute the derivative of y with respect to x via central difference.

    Parameters
    ----------
    x : np.ndarray
        Array of x values
    y : np.ndarray
        Array of y values

    Returns
    -------
    Tuple containing x values and corresponding y derivatives.

    """

    dydx = ((np.roll(y, -1) - np.roll(y, 1)) \
            / (np.roll(x, -1) - np.roll(x, 1)))[1:-1]

    return x[1:-1], dydx