import numpy as np
from scipy import linalg
from constants import EPS
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
    size = m1.shape[0]
    _, u = linalg.eigh(m1 + m2, overwrite_a=True, eigvals=(max(0, size - dim), size - 1), check_finite=False)
    u = np.fliplr(u)
    # return np.conj(u[:, :dim])
    return np.conj(u)


def extend_and_project_transfer_matrix(t4, weight, u1, u2=None):
    """
    Returns extended and re-normalized corner matrix according to following schema:

            a
            |
           / \ conj(U1) or U1
          /   \
         |     |  Weight
      T4 |_____|____
         |     |     j
         |     |
          \   /
           \ /  U1 or U2
            |
            b

    """

    da, di = t4.shape[:2]
    u1 = u1.reshape((da, di, -1))

    if u2 is None:
        u2 = u1
        u1 = np.conj(u1)
    else:
        u2 = u2.reshape((da, di, -1))

    t4u = np.tensordot(t4, u1, axes=(0, 0))  # t4u_{l d i a} = t4_{g l d} * u1_{(g, i) a}
    t4uw = np.tensordot(t4u, weight, axes=([0, 2], [3, 0]))  # t4uw_{d a j k} = t4u_{l d i a} * weight_{i j k l}
    return np.tensordot(t4uw, u2, axes=([0, 3], [0, 1]))  # t4_new_{a j b} = t4uw_{d a j k} * u2_{(d, k), b}


def tuple_rotation(c1, c2, c3, c4):
    """Returns new tuple shifted to left by one place."""

    return c2, c3, c4, c1


def weight_rotate(weight):
    """Returns weight rotated anti-clockwise."""

    return np.transpose(weight, (1, 2, 3, 0))


def orus_vidal_2009(dim, weight, corners, transfer_matrices, rotate=True):
    """Returns corners and transfer matrices extended (and projected) by one iterative CTMRG step. Here, one step of
    CTMRG consists of repeating four times following two steps:

        (1) introducing additional column to system;
        (2) 90-degrees rotation of whole system.

    This algorithm comes from PHYSICAL REVIEW B 80, 094403 (2009).

    """

    c1, c2, c3, c4 = corners
    t1, t2, t3, t4 = transfer_matrices

    for _ in range(4):

        c1 = extend_corner1(c1, t1)
        c4 = extend_corner4(c4, t3)

        u = create_projector_operator(dim, c1, c4)

        c1 = np.tensordot(c1, u, axes=(1, 0))
        c4 = np.tensordot(np.conj(u), c4, axes=(0, 0))

        t4 = extend_and_project_transfer_matrix(t4, weight, u)

        if rotate:
            c1, c2, c3, c4 = tuple_rotation(c1, c2, c3, c4)
            t1, t2, t3, t4 = tuple_rotation(t1, t2, t3, t4)
            weight = weight_rotate(weight)

    return [c1, c2, c3, c4], [t1, t2, t3, t4]


def create_upper_half(c1, c2):
    """

        C1  ____ __......__ ____  C2
           |    |          |    |
           |____|__......__|____|
           |    |          |    |
           |    |          |    |
           a    i          j    b

    """

    uh = np.tensordot(c2, c1, axes=([2, 3], [0, 1]))  # uh_{b j a i} = c2_{b j c k} * c1_{c k a i}
    return uh.reshape((np.prod(uh.shape[:2]), -1))  # uh_{(b j), (a i)}


def create_lower_half(c3, c4):
    """

           a    i              j    b
           |    |              |    |
           |____|____......____|____|
           |    |              |    |
           |____|____......____|____|
        C4                            C3

    """

    lh = np.tensordot(c3, c4, axes=([0, 1], [2, 3]))  # lh_{b j a i} = c3_{c k b j} * c4_{a i c k}
    return lh.reshape((np.prod(lh.shape[:2]), -1))  # lh_{(b j), (a i)}


def create_projectors(c1, c2, c3, c4, dim_cut):
    """Returns upper and lower projector used for renormalization of corners c1, c4 and transfer matrix t4.

    Projector pair is obtained according to schema in Figure 1 in PHYSICAL REVIEW B 84, 041108(R) (2011).

    """

    upper_half = create_upper_half(c1, c2)
    # q, r_up = linalg.qr(upper_half)
    _, r_up = linalg.qr(upper_half)

    lower_half = create_lower_half(c3, c4)
    _, r_down = linalg.qr(lower_half)

    rr = np.tensordot(r_up, r_down, axes=(1, 1))
    u, s, vt = linalg.svd(rr, lapack_driver='gesvd')  # use 'gesvd' or 'gesdd'

    dim_new = min(s.shape[0], dim_cut)
    lambda_new = []

    for x in s[:dim_new]:
        if x < EPS:
            # print('s too small', s)
            # print('s too small')
            break
        lambda_new.append(x)

    dim_new = len(lambda_new)
    lambda_new = np.array(lambda_new)

    u = np.conj(u[:, :dim_new]) / np.sqrt(lambda_new)[None, :]
    vt = np.conj(vt[:dim_new, :]) / np.sqrt(lambda_new)[:, None]

    upper_projector = np.tensordot(r_down, vt, axes=(0, 1))
    lower_projector = np.tensordot(r_up, u, axes=(0, 0))

    return upper_projector, lower_projector


def corboz_at_al_2011(dim, weight, corners, transfer_matrices, rotate=True):
    """Returns corners and transfer matrices extended (and projected) by one iterative CTMRG step. Here, one step of
    CTMRG consists of repeating four times following two steps:

        (1) introducing additional column to system;
        (2) 90-degrees rotation of whole system.

    This algorithm comes from PHYSICAL REVIEW B 84, 041108(R) (2011).

    """

    c1, c2, c3, c4 = corners
    t1, t2, t3, t4 = transfer_matrices

    for _ in range(4):
        corners_extended = []

        for i in range(4):
            weight = weight_rotate(weight)
            c = corners[i]
            tm1 = transfer_matrices[i]
            tm2 = transfer_matrices[(i + 3) % 4]
            # print(f'corner c[{i+1}] extension...')
            corners_extended.append(corner_extension(c, tm1, tm2, weight))
            # print(f'extended corner c[{i+1}] ready')

        p_up, p_down = create_projectors(*corners_extended, dim)

        c1 = extend_corner1(c1, t1)
        c4 = extend_corner4(c4, t3)

        c1 = np.tensordot(c1, p_up, axes=(1, 0))
        c4 = np.tensordot(p_down, c4, axes=(0, 0))

        t4 = extend_and_project_transfer_matrix(t4, weight, p_down, p_up)

        if rotate:
            c1, c2, c3, c4 = tuple_rotation(c1, c2, c3, c4)
            t1, t2, t3, t4 = tuple_rotation(t1, t2, t3, t4)
            transfer_matrices = t1, t2, t3, t4
            corners = c1, c2, c3, c4
            weight = weight_rotate(weight)

    return [c1, c2, c3, c4], [t1, t2, t3, t4]


def create_density_matrix(c1, c2, c3, c4):
    """
    Returns density matrix constructed from four extended corner matrices.

    Note: This approach is not used in the current version due to numerical instabilities when treating quantum systems.

        DM_{a i b j} =

        C1  ____ __ a      b __ ____  C2
           |    |              |    |
           |____|__          __|____|
           |    |   i      j   |    |
           |    |              |    |
           .    .              .    .
           |    |              |    |
           |____|____......____|____|
           |    |              |    |
           |____|____......____|____|
        C4                            C3

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
        # print('p2 not initialized')
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


def calculate_correlation_length(t1, t3, n=5):  # or use (t2, t4)
    """Returns list of n largest correlation lengths."""

    if t1.shape != t3.shape:
        print('t1.shape', t1.shape)
        print('t3.shape', t3.shape)
        raise ValueError

    t1 = np.array(t1, dtype=np.complex64)
    t3 = np.array(t3, dtype=np.complex64)

    tm = np.tensordot(t1, t3, axes=(1, 1))  # tm_{a1 b1 a2 b2} = t1_{a1 i b1} * t3_{a2 i b2}
    tm = np.transpose(tm, (0, 3, 1, 2))  # tm_{a1 b2 b1 a2}
    # tm = np.transpose(tm, (0, 2, 1, 3))  # tm_{a1 a2 b1 a2}

    d = tm.shape[0]
    tm = tm.reshape((d * d, d * d))

    np.save('transfer_matrix.npy', tm)

    w = linalg.eigvals(tm, overwrite_a=True, check_finite=False)
    w_abs = sorted(abs(w), reverse=True)[:(n+1)]

    correlation_length = [1 / np.log(w_abs[0] / w_abs[j+1]) for j in range(n)]

    return correlation_length


class CTMRG(object):
    """An implementation of Corner Transfer Matrix Renormalisation Group (CTMRG) algorithm."""

    def __init__(self, dim, weight, corners, tms, weight_imp, algorithm='Corboz'):
        """Creates a CTMRG object from initial local weights."""

        self.dim = dim  # cut-off for virtual degrees of freedom
        # self.weight = weight
        # self.corners = corners
        # self.tms = tms
        # self.weight_imp = weight_imp

        self.list_of_log_corner_norms = [np.log(np.max(np.abs(x))) for x in corners]
        self.list_of_log_tm_norms = [np.log(np.max(np.abs(x))) for x in tms]

        self.corners = tuple(map(lambda x: x / np.max(np.abs(x)), corners))
        self.tms = tuple(map(lambda x: x / np.max(np.abs(x)), tms))

        weight_norm = np.max(np.abs(weight))
        print('weight norm', weight_norm)
        self.log_weight_norm = np.log(weight_norm)

        self.weight = weight / weight_norm
        self.weight_imp = weight_imp / weight_norm

        self.algorithm = algorithm
        self.iter_counter = 0

        if self.algorithm == 'Corboz':
            self.system_extension_and_projection = corboz_at_al_2011
        elif self.algorithm == 'Orus':
            self.system_extension_and_projection = orus_vidal_2009
        else:
            raise ValueError("algorithm should be either 'Corboz' or 'Orus'")

    def __getattr__(self, temperature):
        """When temperature property does not exist, it is returned as 1."""
        return 1.

    def calculate_trace_ctm4(self):
        c1, c2, c3, c4 = self.corners
        x = np.tensordot(c2, c1, axes=(1, 0))
        x = np.tensordot(c3, x, axes=(1, 0))
        return np.tensordot(c4, x, axes=([1, 0], [0, 1]))

    @property
    def classical_free_energy(self):

        log_corner_norms = self.list_of_log_corner_norms
        log_tm_norms = self.list_of_log_tm_norms

        n = len(log_corner_norms)
        # assert n == len(log_tm_norms)

        # print('n', n)
        # print('iter', self.iter_counter)

        sum_of_log_norms = sum(log_corner_norms)
        sum_of_log_norms += sum(2 * ((n - 1 - i) // 4) * sum(log_tm_norms[i:i+4]) for i in range(0, n, 4))

        w = 4 * self.log_weight_norm * (self.iter_counter ** 2)

        trace_ctm4 = self.calculate_trace_ctm4()

        num_of_sites = 2 * ((2 * (self.iter_counter + 1)) ** 2) + 4 * (self.iter_counter + 1)

        f = - (np.log(trace_ctm4) + sum_of_log_norms + w) / num_of_sites
        return f * self.temperature

    @property
    def fast_free_energy(self):
        # w = self.log_weight_norm * (self.iter_counter / (self.iter_counter + 1)) ** 2
        w = self.log_weight_norm
        f = - w - sum(self.list_of_log_tm_norms[-i] for i in range(1, 5)) / 4
        # return f * self.temperature / (2 + 1 / (self.iter_counter + 1))
        return f * self.temperature / 2

    def ctmrg_extend_and_renormalize(self, num_of_steps=1, dim=None, rotate=True):
        """Performs num_of_steps iterations of ctmrg algorithm for given cut-off dim and returns nothing."""

        if dim is None:
            dim = self.dim

        for _ in range(num_of_steps):
            self.corners, self.tms = self.system_extension_and_projection(dim, self.weight, self.corners,
                                                                          self.tms, rotate)
            for j in range(4):
                corner_norm = np.max(np.abs(self.corners[j]))
                self.corners[j] /= corner_norm
                self.list_of_log_corner_norms.append(np.log(corner_norm))
                tm_norm = np.max(np.abs(self.tms[j]))
                self.tms[j] /= tm_norm
                self.list_of_log_tm_norms.append(np.log(tm_norm))
                # print('corner_norm', corner_norm)
                # print('tm_norm', tm_norm)

        self.iter_counter += num_of_steps

    def ctmrg_iteration(self, num_of_steps=100):
        """Performs at most num_of_steps iterations of ctmrg algorithm and prints energy after each iteration.
        Returns the final energy, "precision", and number of iterations."""

        energy = 0
        energy_mem = -1
        # correlation_length_0 = 0
        # correlation_length_mem_0 = -1

        free_energy = 0
        free_energy_mem = -1

        fast_free_energy = 0
        fast_free_energy_mem = -1

        """
        # Procedure for stabilizing corners and tms: it doesn't seem to help much
        for dimension in range(2, self.dim, 2):
            self.ctmrg_extend_and_renormalize(num_of_steps=20, dim=dimension)
        """

        i = 0
        # for i in range(num_of_steps):  # Don't forget to comment out the incrementation of i in the body
        # while abs(free_energy - free_energy_mem) > 1.E-10 and i < num_of_steps:
        # while abs(fast_free_energy - fast_free_energy_mem) > 1.E-14 and i < num_of_steps:
        while abs(energy - energy_mem) > 1.E-6 and i < num_of_steps:
            # while abs(correlation_length_0 - correlation_length_mem_0) > 1.E-7 and i < num_of_steps:

            self.ctmrg_extend_and_renormalize()

            """
            assert np.allclose(self.corners[0], self.corners[2])
            assert np.allclose(self.corners[1], self.corners[3])
            assert np.allclose(self.tms[0], self.tms[2])
            assert np.allclose(self.tms[1], self.tms[3])
            """

            energy_mem = energy
            energy = measurement(self.weight, self.corners, self.tms, self.weight_imp)

            fast_free_energy_mem = fast_free_energy
            fast_free_energy = self.fast_free_energy

            free_energy_mem = free_energy
            free_energy = self.classical_free_energy

            try:
                # correlation_length_mem_0 = correlation_length_0
                raise ValueError
                correlation_length = calculate_correlation_length(self.tms[1], self.tms[3])
                # correlation_length_0 = correlation_length[0]
                # correlation_length_02 = calculate_correlation_length(self.tms[0], self.tms[2])
            except ValueError:
                correlation_length = [-1] * 5

            # print('ctm iter', i, 3 * energy / 2, *correlation_length)
            # print('ctm iter', i, 3 * energy / 2, correlation_length, correlation_length_02)
            # print(f"""{i + 1}\t{np.real(3 * energy / 2)}\t{' '.join(str(cl) for cl in correlation_length)}""")
            # print('f = ', self.classical_free_energy())
            tab = '\t'
            # print(f"""{i + 1}\t{3 * energy / 2}\t\t{tab.join(str(cl) for cl in correlation_length)}""")
            print(f"{i + 1}\t{energy}\t{free_energy}\t{fast_free_energy}"
                  f"\t\t{tab.join(str(cl) for cl in correlation_length)}")

            i += 1

        return energy, abs(energy - energy_mem), correlation_length, i
