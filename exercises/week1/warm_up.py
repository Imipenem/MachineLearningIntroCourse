# used for manipulating directory paths
import os

# Scientific and vector computation for python
import numpy as np

# Plotting library
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D  # needed to plot 3-D surfaces


def warm_up_exercise():
    """
        Example function in Python which computes the identity matrix.

        Returns
        -------
        A : array_like
            The 5x5 identity matrix.

        Instructions
        ------------
        Return the 5x5 identity matrix.
    """
    A = np.eye(5)
    return A


if __name__ == '__main__':
    print(warm_up_exercise())
