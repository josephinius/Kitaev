import math
import copy
import numpy as np
from scipy import linalg
# import time
# import pickle
from tqdm import tqdm

"""

C_ab:

      a
     |
     |____ b
  
T_aib:

    a
    |
    |_____ i
    |
    |
    b

W_ijkl:

          i
          |
    l ____|____ j
          |
          |
          k


Partition function:
  
  C1  ____.____ C2
     |    .    |
     |. . . . .|
     |    .    |
     |____.____|
  C4            C3


"""


def create_density_matrix(c1, c2, c3, c4):
    """
    Returns density matrix constructed from four extended corner matrices.

      C1  ____ __ a      b __ ____ C2
         |    |              |    |
         |____|__          __|____|
         |    |   i      j   |    |
         |    |              |    |
         .    .              .    .
         |    |              |    |
         |____|____......____|____|
         |    |              |    |
         |____|____......____|____|
      C4                           C3

    """

    c2c3 = np.tensordot(c2, c3, axes=([0, 1], [2, 3]))  # c2c3_{b j d l} = c2_{c k b j} * c3_{d l c k}
    c2c3c4 = np.tensordot(c2c3, c4, axes=([2, 3], [2, 3]))  # c2c3c4_{b j e m} = c2c3_{b j d l} * c4_{e m d l}
    dm = np.tensordot(c1, c2c3c4, axes=([2, 3], [2, 3]))  # dm_{a i b j} = c1_{a i e m} * c2c3c4_{b j e m}
    return dm


def create_projector(density_matrix, dim):
    """Returns projector used for renormalization of corner matrix and transfer matrix."""

    da, di, db, dj = density_matrix.shape
    density_matrix = density_matrix.reshape((da * di, db * dj))
    projector, _, _ = linalg.svd(density_matrix, lapack_driver='gesvd')  # use 'gesvd' or 'gesdd'
    projector = projector[:, :dim].reshape((da, di, -1))
    return projector


def corner_extension(corner, tm1, tm2, weight):
    """
    Returns extended corner matrix according to following schema:

       a    i
       |    |  W
    T1 |____|____ j
       |    |
       |____|____ b
     C      T2

    """

    ct = np.tensordot(tm1, corner, axes=(2, 0))  # ct_{a, l, q} = tm1_{a, l, p} * corner_{p, q}
    ctt = np.tensordot(ct, tm2, axes=(2, 0))  # ctt_{a l k b} = ct_{a, l, q} * tm2_{q k b}
    cttw = np.tensordot(ctt, weight, axes=([1, 2], [3, 2]))  # cttw_{a b i j} = ctt_{a l k b} * weight_{i j k l}
    return np.swapaxes(cttw, 1, 2)  # np.transpose(cttw, (0, 2, 1, 3))  # cttw_{a i b j}


def transfer_matrix_extension(tm, weight):
    """
    Returns extended corner matrix according to following schema:

        a     i
        |     |
     T  |_____|____ j
        |     |
        |     |
        b     k

    """

    tw = np.tensordot(tm, weight, axes=(1, 3))  # tw_{a b i j k} = tm_{a l b} * weight_{i j k l}
    return np.transpose(tw, (0, 2, 3, 1, 4))  # tw_{a i j b k}


def corner_renormalization(corner, p1, p2):
    """
    Returns re-normalized corner matrix.

          a
          |
         / \  P1
        /   \
       |____|___
       |    |   \___ b
       |____|___/
     C           P2

    """

    cp1 = np.tensordot(p1, corner, axes=([0, 1], [0, 1]))  # cp1_{a t j} = p1_{s i a} * corner_{s i t j}
    projected_corner = np.tensordot(cp1, p2, axes=([1, 2], [0, 1]))  # cp1p2_{a b} = cp1_{a t j} * p2_{t j b}
    return projected_corner


def transfer_matrix_renormalization(tm, p):
    """
    Returns re-normalized transfer matrix.

           a
           |
          / \  P
         /   \
        |     |
     T  |_____|____ j
        |     |
         \   /
          \ /  P
           |
           b

    """

    tmp = np.tensordot(p, tm, axes=([0, 1], [0, 1]))  # tmp_{a j t k} = p_{s i a} * tm_{s i j t k}
    projected_tm = np.tensordot(tmp, p, axes=([2, 3], [0, 1]))  # projected_tm_{a j b} = tmp_{a j t k} * p_{t k b}
    return projected_tm


def tuple_rotation(c1, c2, c3, c4):
    return c2, c3, c4, c1


def create_four_projectors(c1, c2, c3, c4, dim):
    """Returns projectors for four different cuts of the system.

        projectors = [p1, p2, p3, p4]

                 (1)
          C1  ____.____ C2
             |    .    |
         (4) |. . . . .| (2)
             |    .    |
             |____.____|
          C4     (3)    C3

    """

    projectors = []
    for _ in range(4):
        dm = create_density_matrix(c1, c2, c3, c4)
        dim_a = c1.shape[0]
        dim_i = c1.shape[1]
        assert dim_a == c2.shape[0]
        assert dim_i == c2.shape[1]
        projectors.append(create_projector(dm, dim=min(dim, dim_a * dim_i)))
        c1, c2, c3, c4 = tuple_rotation(c1, c2, c3, c4)
    return projectors


def weight_rotate(weight):
    """Returns weight rotated anti-clockwise."""

    return np.transpose(weight, (1, 2, 3, 0))


def system_extension_and_projection(dim, weight, corners, transfer_matrices):
    """Returns corners and transfer matrices extended (and projected) by one iterative CTMRG step."""

    corners_extended = []
    tms_extended = []

    for i in range(4):
        weight = weight_rotate(weight)
        c = corners[i]
        t1 = transfer_matrices[i % 4]
        t2 = transfer_matrices[(i + 3) % 4]
        corners_extended.append(corner_extension(c, t1, t2, weight))
        tms_extended.append(transfer_matrix_extension(t1, weight))

    projectors = create_four_projectors(*corners_extended, dim)

    corners_projected = []
    tms_projected = []

    for i in range(4):
        p1 = projectors[i % 4]
        p2 = projectors[(i + 3) % 4]
        corners_projected.append(corner_renormalization(corners_extended[i], p1, p2))
        tms_projected.append(transfer_matrix_renormalization(transfer_matrices[i % 4], p1))

    return corners_projected, tms_projected


def measurement(weight, corners, transfer_matrices, weight_imp):
    """
    Returns expectation value for weight impurity in the center of system.

                    T1
        C1  ____ _______ ____ C2
           |        |        |
           |        |        |
      T4   |________O________|   T2
           |        |        |
           |        |        |
           |________|________|
        C4                    C3
                    T3

    """

    c1, c2, c3, c4 = corners
    t1, t2, t3, t4 = transfer_matrices

    c1t1 = np.tensordot(c1, t1, axes=(0, 2))  # c1t1_{h b i} = c1_{a h} * t1_{b i a}
    c1t1c2 = np.tensordot(c1t1, c2, axes=(1, 0))  # c1t1c2_{h i c} = c1t1_{h b i} * c2_{c b}
    half1 = np.tensordot(c1t1c2, t2, axes=(2, 2))  # half1_{h i d j} = c1t1c2_{h i c} * t2_{d j c}

    t4c4 = np.tensordot(t4, c4, axes=(2, 0))  # t4c4_{h l f} = t4_{h l g} * c4_{g f}
    t4c4t3 = np.tensordot(t4c4, t3, axes=(2, 0))  # t4c4t3_{h l k e} = t4c4_{h l f}  * t3_{f k e}
    t4c4t3c3 = np.tensordot(t4c4t3, c3, axes=(3, 0))  # t4c4t3c3_{h l k d} = t4c4t3_{h l k e} * c3_{e d}

    half2 = np.tensordot(t4c4t3c3, weight, axes=([1, 2], [3, 2]))  # half2_{h d i j} = t4c4t3c3_{h l k d} * w_{i j k l}
    norm = np.tensordot(half1, half2, axes=([0, 1, 2, 3], [0, 2, 1, 3]))
    half2 = np.tensordot(t4c4t3c3, weight_imp, axes=([1, 2], [3, 2]))

    return np.tensordot(half1, half2, axes=([0, 1, 2, 3], [0, 2, 1, 3])) / norm


class CTMRG(object):

    def __init__(self, dim, weight, corners, tms, weight_imp):

        self.dim = dim
        self.weight = weight
        self.corners = corners
        self.tms = tms
        self.weight_imp = weight_imp

        self.iter_counter = 0

    def ctmrg_iteration(self, num_of_steps):

        for i in range(num_of_steps):
            self.corners, self.tms = system_extension_and_projection(self.dim, self.weight, self.corners, self.tms)
            for j in range(4):
                self.corners[j] /= np.max(self.corners[j])
                self.tms[j] /= np.max(self.tms[j])

            energy = measurement(self.weight, self.corners, self.tms, self.weight_imp)
            print(i, energy)
