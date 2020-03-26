import numpy as np
from scipy import linalg
# import math
# import copy
# import time
# import pickle
# from tqdm import tqdm


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


def extend_corner1(c1, t1):
    """
    Returns extended corner matrix c1 according to following schema:

                     T1
        C1  ____.____ ____ a
           |         |
           |         |
         ( b         i )

    """

    c1t1 = np.tensordot(c1, t1, axes=(0, 2))  # c1t1_{b a i} = c1_{g b} * t1_{a i g}
    c1t1 = np.transpose(c1t1, (1, 0, 2))  # c1t1_{a b i}
    return c1t1.reshape((t1.shape[0], -1))  # c1t1_{a (b i)}


def extend_corner4(c4, t3):
    """
    Returns extended corner matrix c4 according to following schema:

         ( a         i )
           |         |
           |____.____|____ b
        C4           T3

    """

    c4t3 = np.tensordot(c4, t3, axes=(1, 0))  # c4t3_{a i b} = c4_{a g} * t3_{g i b}
    return c4t3.reshape((-1, t3.shape[2]))  # c4t3_{(a i) b}


def create_projector_operator(dim, c1, c4):
    """Returns (complex conjugated) projector used for renormalization of corner matrix and transfer matrix.
    Projector is obtained by eigenvalue decomposition of m1 + m2 (see Figure 2 in PHYSICAL REVIEW B 80, 094403 (2009)).


    m1_{(a i), (b j)} =

        C1_extended     conj(C1_extended)
            ____ ____...____ ____
           |    |           |    |
           |    |           |    |
         ( a    i )       ( j    b )

    m2_{(a i), (b j)} =

        C4_extended     conj(C4_extended)
            ____ ____...____ ____
           |    |           |    |
           |    |           |    |
         ( a    i )       ( j    b )

    """

    m1 = np.tensordot(c1, np.conj(c1), axes=(0, 0))  # m1_{ai bj} = c1_{g (a i)} * conj(c1)_{g (b j)}
    m2 = np.tensordot(c4, np.conj(c4), axes=(1, 1))  # m2_{ai bj} = c4_{(a i) g} * conj(c4)_{(b j) g}

    # w, u = np.linalg.eigh(m1 + m2)
    # w, u = linalg.eigh(m1 + m2, overwrite_a=True, check_finite=False)
    _, u = linalg.eigh(m1 + m2, overwrite_a=True, check_finite=False)
    u = np.fliplr(u)

    return np.conj(u[:, :dim])


def extend_and_project_transfer_matrix(t4, weight, u):
    da, di = t4.shape[:2]
    u = u.reshape((da, di, -1))
    t4u = np.tensordot(t4, np.conj(u), axes=(0, 0))  # t4u_{l d i a} = t4_{g l d} * u_{(g, i) a}
    t4uw = np.tensordot(t4u, weight, axes=([0, 2], [3, 0]))  # t4uw_{d a j k} = t4u_{l d i a} * weight_{i j k l}
    return np.tensordot(t4uw, u, axes=([0, 3], [0, 1]))  # t4_new_{a j b} = t4uw_{d a j k} * u_{(d, k), b}


def tuple_rotation(c1, c2, c3, c4):
    return c2, c3, c4, c1


def weight_rotate(weight):
    """Returns weight rotated anti-clockwise."""

    return np.transpose(weight, (1, 2, 3, 0))


def system_extension_and_projection(dim, weight, corners, transfer_matrices):

    c1, c2, c3, c4 = corners
    t1, t2, t3, t4 = transfer_matrices

    for _ in range(4):

        c1 = extend_corner1(c1, t1)
        c4 = extend_corner4(c4, t3)

        u = create_projector_operator(dim, c1, c4)

        c1 = np.tensordot(c1, u, axes=(1, 0))
        c4 = np.tensordot(np.conj(u), c4, axes=(0, 0))

        t4 = extend_and_project_transfer_matrix(t4, weight, u)

        c1, c2, c3, c4 = tuple_rotation(c1, c2, c3, c4)
        t1, t2, t3, t4 = tuple_rotation(t1, t2, t3, t4)
        weight = weight_rotate(weight)

    return [c1, c2, c3, c4], [t1, t2, t3, t4]


def create_density_matrix(c1, c2, c3, c4):
    """
    Returns density matrix constructed from four extended corner matrices.

    Note: This approach is not used in the current version due to numerical instabilities when treating quantum systems.

        DM_{a i b j} =

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
    return np.tensordot(c1, c2c3c4, axes=([2, 3], [2, 3]))  # dm_{a i b j} = c1_{a i e m} * c2c3c4_{b j e m}


def create_projector(density_matrix, dim):
    """Returns projector used for renormalization of corner matrix and transfer matrix."""

    da, di, db, dj = density_matrix.shape
    # print('dimension is:', da * di)
    # return np.eye(da * di, db * dj).reshape((da, di, db * dj))
    density_matrix = density_matrix.reshape((da * di, db * dj))
    # projector, sv, _ = linalg.svd(density_matrix, lapack_driver='gesvd')  # use 'gesvd' or 'gesdd'
    # w, projector = np.linalg.eigh(density_matrix)
    _, projector = np.linalg.eigh(density_matrix)
    # print('w', w)
    # print('singular values', sv[:dim])
    projector = np.fliplr(projector)
    return np.conj(projector[:, :dim].reshape((da, di, -1)))


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


def transfer_matrix_renormalization(tm, p1, p2=None):
    """
    Returns re-normalized transfer matrix.

           a
           |
          / \  P1
         /   \
        |     |
     T  |_____|____ j
        |     |
         \   /
          \ /  P2
           |
           b

    """

    if p2 is None:
        print('p2 not initialized')
        # exit()
        p2 = p1

    # print('p.shape', p.shape)
    # print('tm.shape', tm.shape)

    tmp = np.tensordot(p1, tm, axes=([0, 1], [0, 1]))  # tmp_{a j t k} = p_{s i a} * tm_{s i j t k}
    projected_tm = np.tensordot(tmp, p2, axes=([2, 3], [0, 1]))  # projected_tm_{a j b} = tmp_{a j t k} * p_{t k b}
    return projected_tm


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
        assert c1.shape[0] == c2.shape[2]
        assert c1.shape[1] == c2.shape[3]
        dm = create_density_matrix(c1, c2, c3, c4)
        # dim_a = c1.shape[0]
        # dim_i = c1.shape[1]
        # projectors.append(create_projector(dm, dim=min(dim, dim_a * dim_i)))
        projectors.append(create_projector(dm, dim))
        c1, c2, c3, c4 = tuple_rotation(c1, c2, c3, c4)

    return projectors


def system_extension_and_projection_old(dim, weight, corners, transfer_matrices):
    """Returns corners and transfer matrices extended (and projected) by one iterative CTMRG step."""

    corners_extended = []
    tms_extended = []

    for i in range(4):
        weight = weight_rotate(weight)
        c = corners[i]
        t1 = transfer_matrices[i]
        t2 = transfer_matrices[(i + 3) % 4]
        # print('corner extension...')
        corners_extended.append(corner_extension(c, t1, t2, weight))
        # print('extended corner ready!')
        # print('tm extension...')
        tms_extended.append(transfer_matrix_extension(t1, weight))
        # print('tm extended ready!')

    # print('create projectors...')
    projectors = create_four_projectors(*corners_extended, dim)
    # print('projectors ready!')

    corners_projected = []
    tms_projected = []

    for i in range(4):
        p1 = projectors[i]
        p2 = projectors[(i + 3) % 4]
        corners_projected.append(corner_renormalization(corners_extended[i], p1, np.conj(p2)))
        tms_projected.append(transfer_matrix_renormalization(tms_extended[i], p1, np.conj(p1)))

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
    c1t1c2 = np.tensordot(c1t1, c2, axes=(1, 1))  # c1t1c2_{h i c} = c1t1_{h b i} * c2_{c b}
    half1 = np.tensordot(c1t1c2, t2, axes=(2, 2))  # half1_{h i d j} = c1t1c2_{h i c} * t2_{d j c}

    t4c4 = np.tensordot(t4, c4, axes=(2, 0))  # t4c4_{h l f} = t4_{h l g} * c4_{g f}
    t4c4t3 = np.tensordot(t4c4, t3, axes=(2, 0))  # t4c4t3_{h l k e} = t4c4_{h l f}  * t3_{f k e}
    t4c4t3c3 = np.tensordot(t4c4t3, c3, axes=(3, 0))  # t4c4t3c3_{h l k d} = t4c4t3_{h l k e} * c3_{e d}

    half2 = np.tensordot(t4c4t3c3, weight, axes=([1, 2], [3, 2]))  # half2_{h d i j} = t4c4t3c3_{h l k d} * w_{i j k l}
    norm = np.tensordot(half1, half2, axes=([0, 1, 2, 3], [0, 2, 1, 3]))

    # print('z', norm)

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

    def ctmrg_iteration(self, num_of_steps=20):

        energy = 0
        energy_mem = -1
        i = 0
        # for i in range(num_of_steps):
        while abs(energy - energy_mem) > 1.E-8 and i < num_of_steps:

            self.corners, self.tms = system_extension_and_projection(self.dim, self.weight, self.corners, self.tms)

            for j in range(4):
                corner_norm = np.max(np.abs(self.corners[j]))
                self.corners[j] /= corner_norm
                # print('corner_norm', corner_norm)
                tm_norm = np.max(np.abs(self.tms[j]))
                self.tms[j] /= tm_norm
                # print('tm_norm', tm_norm)

            """
            assert np.allclose(self.corners[0], self.corners[2])
            assert np.allclose(self.corners[1], self.corners[3])
            assert np.allclose(self.tms[0], self.tms[2])
            assert np.allclose(self.tms[1], self.tms[3])
            """

            energy_mem = energy
            energy = measurement(self.weight, self.corners, self.tms, self.weight_imp)
            print('ctm iter', i, 3 * energy / 2)
            i += 1

        return energy, abs(energy - energy_mem), i
