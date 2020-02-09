import numpy as np

from math import *


class MatrixUtils:

    @staticmethod
    def construct_rotation_matrix(angle, direction, point=None):
        """
        Retrieved from: https://www.lfd.uci.edu/~gohlke/code/transformations.py.html
        """

        sina = sin(angle)
        cosa = cos(angle)
        direction = direction[:3] / np.linalg.norm(direction)

        R = np.diag([cosa, cosa, cosa])
        R += np.outer(direction, direction) * (1.0 - cosa)
        direction *= sina
        R += np.array([[0.0, -direction[2], direction[1]],
                       [direction[2], 0.0, -direction[0]],
                       [-direction[1], direction[0], 0.0]])
        return R
