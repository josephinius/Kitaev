# import math
import numpy as np
from scipy import linalg
import functools
import constants
import copy
# from sklearn.utils.extmath import randomized_svd

EPS = constants.EPS


def tensor_rotate(ten):
    return np.transpose(ten, (0, 2, 3, 1))


def lambdas_rotate(lam):
    return lam[1:] + [lam[0]]


def create_double_tensor(ten, lambdas):
    ten = ten * np.sqrt(lambdas[0])[None, :, None, None]
    ten = ten * np.sqrt(lambdas[1])[None, None, :, None]
    ten = ten * np.sqrt(lambdas[2])[None, None, None, :]

    double_ten = np.einsum('m x y z, m p q r->x p y q z r', ten, np.conj(ten))

    dx, dy, dz = ten.shape[1:]
    return double_ten.reshape((dx * dx, dy * dy, dz * dz))


def deform_and_renorm_double_tensor(dten_a, dten_b, dim_cut):
    # 1) contraction

    """
       z1          y2
        \         /
         \___x___/  T_b
    T_a  /       \
        /         \
       y1         z2

    """

    # print("pair construction...")
    pair = np.tensordot(dten_a, dten_b, axes=(0, 0))  # pair_{y1 z1 y2 z2} = dten_a_{x y1 z1} * dten_b_{x y2 z2}

    return deformation(pair, dim_cut)


def deformation(pair, dim_cut):
    # 2) deformation

    """
       z1    y2
        \   /
         \ /
          |
          |
         / \
        /   \
       y1   z2

    """

    # print('in deformation...')
    # print(pair.shape)

    # pair = np.transpose(pair, (2, 1, 0, 3))  # pair_{y2 z1 y1 z2}
    pair = np.transpose(pair, (1, 2, 3, 0))  # pair_{z1 y2 z2 y1}

    d1, d2, d3, d4 = pair.shape
    pair = pair.reshape((d1 * d2, d3 * d4))

    # 3) SVD
    print("SVD... (honeycomb expectation)")
    x, ss, y = linalg.svd(pair, lapack_driver='gesdd')  # use 'gesvd' or 'gesdd'
    # x, ss, y = linalg.svd(pair, lapack_driver='gesvd')  # use 'gesvd' or 'gesdd'
    # x, ss, y = randomized_svd(pair, n_components=120, n_iter=5, random_state=None)
    print("SVD done")
    # print('ss', ss)

    dim_new = min(ss.shape[0], dim_cut)

    lambda_new = []
    for s in ss[:dim_new]:
        if s < EPS:
            print('s too small', s)
            # print('ss[:dim_new]', ss[:dim_new])
            break
        lambda_new.append(s)

    dim_new = len(lambda_new)
    lambda_new = np.array(lambda_new)
    # lambda_new = lambda_new / calculate_norm(lambda_new)

    # print('lambda_new', lambda_new)

    x = x[:, :dim_new]
    y = y[:dim_new, :]

    x = x.reshape((d1, d2, dim_new))
    x = x.transpose((2, 0, 1))

    y = y.reshape((dim_new, d3, d4))
    # y = y.transpose((0, 1, 2))

    x *= np.sqrt(lambda_new)[:, None, None]  # S_B
    y *= np.sqrt(lambda_new)[:, None, None]  # S_A

    # return x, y, lambda_new
    return y, x, lambda_new


def create_deformed_tensors(double_tensor_a, double_tensor_b, dim_cut):
    deformed_tensors = [None] * 3

    assert type(double_tensor_a) == type(double_tensor_b)

    if isinstance(double_tensor_a, tuple):
        assert len(double_tensor_a) == len(double_tensor_b) == 3

        double_tensor_a = list(double_tensor_a)
        double_tensor_b = list(double_tensor_b)

        for i in range(3):
            deformed_tensors[i] = deform_and_renorm_double_tensor(double_tensor_a[i], double_tensor_b[i], dim_cut)
            for j in range(3):
                double_tensor_a[j] = np.transpose(double_tensor_a[j], (1, 2, 0))
                double_tensor_b[j] = np.transpose(double_tensor_b[j], (1, 2, 0))
    else:
        for i in range(3):
            deformed_tensors[i] = deform_and_renorm_double_tensor(double_tensor_a, double_tensor_b, dim_cut)
            double_tensor_a = np.transpose(double_tensor_a, (1, 2, 0))
            double_tensor_b = np.transpose(double_tensor_b, (1, 2, 0))

    sx_a, sx_b, _ = deformed_tensors[0]
    deformed_tensors[0] = (sx_a, sx_b)

    sy_a, sy_b, _ = deformed_tensors[1]
    # sy_a = np.transpose(sy_a, (2, 0, 1))
    # sy_b = np.transpose(sy_b, (2, 0, 1))
    sy_a = np.transpose(sy_a, (1, 2, 0))
    sy_b = np.transpose(sy_b, (1, 2, 0))
    deformed_tensors[1] = (sy_a, sy_b)

    sz_a, sz_b, _ = deformed_tensors[2]
    # sz_a = np.transpose(sz_a, (1, 2, 0))
    # sz_b = np.transpose(sz_b, (1, 2, 0))
    sz_a = np.transpose(sz_a, (2, 0, 1))
    sz_b = np.transpose(sz_b, (2, 0, 1))
    deformed_tensors[2] = (sz_a, sz_b)

    return deformed_tensors


def update_double_tensor(ds_x, ds_y, ds_z):
    """
                   x
                   |
                   |(dsx)
                  / \
               j /   \ k
                /     \
          z____/_ _ _ _\____y
            (ds_z)  i  (ds_y)

    """

    # print('ds_x.shape', ds_x.shape)
    # print('ds_y.shape', ds_y.shape)
    # print('ds_z.shape', ds_z.shape)

    temp = np.tensordot(ds_x, ds_y, axes=(1, 1))  # temp_{x j i y} = dsx_{x k j} dsy_{i k y}
    # temp = np.tensordot(ds_x, ds_y, axes=(2, 2))  # temp_{x j i y} = dsx_{x j k} dsy_{i y k}
    result = np.tensordot(temp, ds_z, axes=([1, 2], [2, 0]))  # result_{x y z} = temp_{x j i y} dsz_{i z j}
    # result = np.tensordot(temp, ds_z, axes=([1, 2], [1, 0]))  # result_{x y z} = temp_{x j i y} dsz_{i j z}

    return result


def create_plaquette(ten_a, ten_b, ten_c, ten_d, ten_e, ten_f):
    FA = np.tensordot(ten_f, ten_a, axes=(1, 1))  # FA_{x z2 x1 z} = F_{x y1 z2} A_{x1 y1 z}
    FAB = np.tensordot(FA, ten_b, axes=(2, 0))  # FAB_{x z2 z y z1} = FA_{x z2 x1 z} B_{x1 y z1}
    FA = None

    ED = np.tensordot(ten_e, ten_d, axes=(0, 0))  # ED_{yy z2 y2 zz} = E_{x2 yy z2} D_{x2 y2 zz}
    EDC = np.tensordot(ED, ten_c, axes=(2, 1))  # EDC_{yy z2 zz xx z1} = ED_{yy z2 y2 zz} C_{xx y2 z1}
    ED = None

    # plaqette_{x z y yy zz xx} = FAB_{x z2 z y z1} EDC_{yy z2 zz xx z1}
    plaquette = np.tensordot(FAB, EDC, axes=([1, 4], [1, 4]))

    FAB = None
    EDC = None

    return np.transpose(plaquette, (0, 5, 2, 3, 1, 4))  # plaquette_{x xx y yy z zz}


def partition_function(ten_a, ten_b, ten_c, ten_d, ten_e, ten_f):
    """

    Directions:
        * x : -
        * y : /
        * z : \

          \        /
           \ A    /
            O----O B
         F /      \
       ---O        O---
           \      / C
          E O----O
           /    D \
          /        \

    """

    AD = np.tensordot(ten_a, ten_d, axes=(2, 2))  # AD_{x1 y1 x2 y2} = A_{x1 y1 z} D_{x2 y2 z}
    BE = np.tensordot(ten_b, ten_e, axes=(1, 1))  # BE_{x1 z1 x2 z2} = B_{x1 y z1} E_{x2 y z2}
    ABED = np.tensordot(AD, BE, axes=([0, 2], [0, 2]))  # ABED_{y1 y2 z1 z2} = AD_{x1 y1 x2 y2} BE_{x1 z1 x2 z2}
    FC = np.tensordot(ten_f, ten_c, axes=(0, 0))  # FC_{y1 z2 y2 z1} = F_{x y1 z2} C_{x y2 z1}
    z = np.tensordot(ABED, FC, axes=([0, 1, 2, 3], [0, 2, 3, 1]))  # z (scalar) = ABED_{y1 y2 z1 z2} FC_{y1 z2 y2 z1}

    """
    AD = np.tensordot(ten_a, ten_d, axes=(0, 0))  # AD_{y1 z1 y2 z2} = A_{x y1 z1} D_{x y2 z2}
    BE = np.tensordot(ten_b, ten_e, axes=(2, 2))  # BE_{x1 y1 x2 y2} = B_{x1 y1 z} E_{x2 y2 z}
    ABED = np.tensordot(AD, BE, axes=([0, 2], [1, 3]))  # ABED_{z1 z2 x1 x2} = AD_{y1 z1 y2 z2} BE_{x1 y1 x2 y2}
    FC = np.tensordot(ten_f, ten_c, axes=(1, 1))  # FC_{x2 z1 x1 z2} = F_{x2 y z1} C_{x1 y z2}
    z = np.tensordot(ABED, FC, axes=([0, 1, 2, 3], [3, 0, 2, 1]))  # z (scalar) = ABED_{z1 z2 x1 x2} FC_{x2 z1 x1 z2}
    """

    return z


def create_double_impurity(ten, lambdas, operator):

    """
    heisenberg = - np.kron(sx, sx) / 2 +
                 - np.kron(sy, sy) / 2 +
                 - np.kron(sz, sz) / 2 +
    """

    ten = ten * np.sqrt(lambdas[0])[None, :, None, None]
    ten = ten * np.sqrt(lambdas[1])[None, None, :, None]
    ten = ten * np.sqrt(lambdas[2])[None, None, None, :]

    dx, dy, dz = ten.shape[1:]

    # assert d == ten.shape[0]
    assert lambdas[0].shape[0] == dx
    assert lambdas[1].shape[0] == dy
    assert lambdas[2].shape[0] == dz

    if isinstance(operator, tuple):
        # for Heisenberg model
        assert len(operator) == 3
        exit()
        result = [None] * 3
        for i in range(3):
            ten_sigma = np.tensordot(operator[i], ten, axes=(1, 0))  # ten_sigma_{i x y z} = sigma_{i j} ten_{j x y z}
            double = np.einsum('m x y z, m p q r->x p y q z r', ten_sigma, np.conj(ten))
            result[i] = double.reshape((dx * dx, dy * dy, dz * dz))
    else:
        # for Kitaev model
        ten_sigma = np.tensordot(operator, ten, axes=(1, 0))  # ten_sigma_{i x y z} = sigma_{i j} ten_{j x y z}
        double = np.einsum('m x y z, m p q r->x p y q z r', ten_sigma, np.conj(ten))
        result = double.reshape((dx * dx, dy * dy, dz * dz))

    return result


def deform_and_renorm_impurity(dten_a, dten_b, dim_cut):
    # print("in deform_and_renorm_impurity")
    assert type(dten_a) == type(dten_b)
    # pair_{y1 z1 y2 z2} = dten_a_{x y1 z1} * dten_b_{x y2 z2}
    if isinstance(dten_a, list):
        print('isinstance list')
        # exit()
        assert len(dten_a) == len(dten_b) == 3
        pair = functools.reduce(lambda x, y: x + y, (np.tensordot(a, b, axes=(0, 0)) for a, b in zip(dten_a, dten_b)))
    else:
        pair = np.tensordot(dten_a, dten_b, axes=(0, 0))
    # print(pair.shape)
    return deformation(pair, dim_cut)


def create_deformed_12ring(double_impurity_6ring, double_tensor_a, double_tensor_b, dim_cut):
    # double_impurity_6ring = A, B, C, D, E, F

    A, B, C, D, E, F = double_impurity_6ring

    result = []  # = [None] * 12  # a - a', b - b', c - c', d - d', e - e', f - f'

    rot_x = rot_neg_x = lambda ten3: ten3  # rotate by 0
    rot_y = rot_neg_z = lambda ten3: np.transpose(ten3, (1, 2, 0))  # rotate by 1
    rot_z = rot_neg_y = lambda ten3: np.transpose(ten3, (2, 0, 1))  # rotate by 2

    # TODO: simplify the expressions below into one simple loop

    ring_dim_cut = dim_cut

    # deform_and_renorm_double_tensor = deform_and_renorm_impurity

    a, ap, _ = deform_and_renorm_double_tensor(*map(rot_z, (A, double_tensor_b)), dim_cut=ring_dim_cut)
    # result.extend(list(map(rot_neg_z, (a, ap))))
    result.extend(list(map(rot_z, (a, ap))))

    b, bp, _ = deform_and_renorm_double_tensor(*map(rot_y, (double_tensor_a, B)), dim_cut=ring_dim_cut)
    # result.extend(list(map(rot_neg_y, (b, bp))))
    result.extend(list(map(rot_y, (b, bp))))

    c, cp, _ = deform_and_renorm_double_tensor(*map(rot_x, (C, double_tensor_b)), dim_cut=ring_dim_cut)
    # result.extend(list(map(rot_neg_x, (c, cp))))
    result.extend(list(map(rot_x, (c, cp))))

    d, dp, _ = deform_and_renorm_double_tensor(*map(rot_z, (double_tensor_a, D)), dim_cut=ring_dim_cut)
    # result.extend(list(map(rot_neg_z, (d, dp))))
    result.extend(list(map(rot_z, (d, dp))))

    e, ep, _ = deform_and_renorm_double_tensor(*map(rot_y, (E, double_tensor_b)), dim_cut=ring_dim_cut)
    # result.extend(list(map(rot_neg_y, (e, ep))))
    result.extend(list(map(rot_y, (e, ep))))

    f, fp, _ = deform_and_renorm_double_tensor(*map(rot_x, (double_tensor_a, F)), dim_cut=ring_dim_cut)
    # result.extend(list(map(rot_neg_x, (f, fp))))
    result.extend(list(map(rot_x, (f, fp))))

    return result


def update_6ring(deformed_12ring, deformed_tensors):
    a, ap, b, bp, c, cp, d, dp, e, ep, f, fp = deformed_12ring

    # result = []  # = [None] * 6  # A, B, C, D, E, F

    # TODO: simplify the expressions below into one simple loop

    A = update_double_tensor(deformed_tensors[0][0], b, a)
    B = update_double_tensor(cp, bp, deformed_tensors[2][1])
    C = update_double_tensor(c, deformed_tensors[1][0], d)
    D = update_double_tensor(deformed_tensors[0][1], ep, dp)
    E = update_double_tensor(f, e, deformed_tensors[2][0])
    F = update_double_tensor(fp, deformed_tensors[1][1], ap)

    return [A, B, C, D, E, F]


def energy_six_directions(double_tensor_a, double_tensor_b, double_impurity_tensors, num_of_iter):

    ten_a = ten_c = ten_e = double_tensor_a
    ten_b = ten_d = ten_f = double_tensor_b
    ten_a = double_impurity_tensors[0][0]  # dimp_ten_a
    ten_b = double_impurity_tensors[0][1]  # dimp_ten_b
    ox1 = partition_function(ten_a, ten_b, ten_c, ten_d, ten_e, ten_f)
    # O = create_plaquette(ten_a, ten_b, ten_c, ten_d, ten_e, ten_f)

    ten_a = ten_c = ten_e = double_tensor_a
    ten_b = ten_d = ten_f = double_tensor_b
    ten_d = double_impurity_tensors[0][1]
    ten_e = double_impurity_tensors[0][0]
    ox2 = partition_function(ten_a, ten_b, ten_c, ten_d, ten_e, ten_f)
    # O += create_plaquette(ten_a, ten_b, ten_c, ten_d, ten_e, ten_f)

    ten_a = ten_c = ten_e = double_tensor_a
    ten_b = ten_d = ten_f = double_tensor_b
    ten_a = double_impurity_tensors[1][0]
    ten_f = double_impurity_tensors[1][1]
    oy1 = partition_function(ten_a, ten_b, ten_c, ten_d, ten_e, ten_f)
    # O += create_plaquette(ten_a, ten_b, ten_c, ten_d, ten_e, ten_f)

    ten_a = ten_c = ten_e = double_tensor_a
    ten_b = ten_d = ten_f = double_tensor_b
    ten_c = double_impurity_tensors[1][0]
    ten_d = double_impurity_tensors[1][1]
    oy2 = partition_function(ten_a, ten_b, ten_c, ten_d, ten_e, ten_f)
    # O += create_plaquette(ten_a, ten_b, ten_c, ten_d, ten_e, ten_f)

    ten_a = ten_c = ten_e = double_tensor_a
    ten_b = ten_d = ten_f = double_tensor_b
    ten_b = double_impurity_tensors[2][1]
    ten_c = double_impurity_tensors[2][0]
    oz1 = partition_function(ten_a, ten_b, ten_c, ten_d, ten_e, ten_f)
    # O += create_plaquette(ten_a, ten_b, ten_c, ten_d, ten_e, ten_f)

    ten_a = ten_c = ten_e = double_tensor_a
    ten_b = ten_d = ten_f = double_tensor_b
    ten_e = double_impurity_tensors[2][0]
    ten_f = double_impurity_tensors[2][1]
    oz2 = partition_function(ten_a, ten_b, ten_c, ten_d, ten_e, ten_f)
    # O += create_plaquette(ten_a, ten_b, ten_c, ten_d, ten_e, ten_f)

    # calculation of the norm
    ten_a = ten_c = ten_e = double_tensor_a
    ten_b = ten_d = ten_f = double_tensor_b
    norm = partition_function(ten_a, ten_b, ten_c, ten_d, ten_e, ten_f)

    print('partition function', norm)

    # o = px1 + px2 + py1 + py2 + pz1 + pz2
    # O = np.einsum('x x y y z z->', O)

    if num_of_iter % 1 == 0:
        print('Expect. iter x1:', num_of_iter, 'energy:', - 3 * (ox1 / norm) / 2)
        print('Expect. iter x2:', num_of_iter, 'energy:', - 3 * (ox2 / norm) / 2)
        print('Expect. iter y1:', num_of_iter, 'energy:', - 3 * (oy1 / norm) / 2)
        print('Expect. iter y2:', num_of_iter, 'energy:', - 3 * (oy2 / norm) / 2)
        print('Expect. iter z1:', num_of_iter, 'energy:', - 3 * (oz1 / norm) / 2)
        print('Expect. iter z2:', num_of_iter, 'energy:', - 3 * (oz2 / norm) / 2)

    return [ox1, ox2, oy1, oy2, oz1, oz2] / norm


def coarse_graining_procedure(tensor_a, tensor_b, lambdas, D):

    """
    Returns the converged energy given the quantum state (tensor_a, tensor_b) for the spin={1/2, 1} Kitaev model and
    prints log of the convergence wrt iterative steps of coarse-graining.
    """

    dim_cut = D * D
    d = tensor_a.shape[0]  # physical dimension

    spin = None
    if d == 2:
        spin = "1/2"
    if d == 3:
        spin = "1"

    # norm_a = np.max(np.abs(tensor_a))
    # norm_b = np.max(np.abs(tensor_b))
    # tensor_a /= norm_a
    # tensor_b /= norm_b

    double_tensor_a = create_double_tensor(tensor_a, lambdas)
    double_tensor_b = create_double_tensor(tensor_b, lambdas)

    print('double_tensor_a.shape', double_tensor_a.shape)
    print('double_tensor_b.shape', double_tensor_b.shape)

    # print('double_tensor_a', double_tensor_a)
    # print('double_tensor_b', double_tensor_b)

    """
    dimp_ten_a_init = create_double_impurity(tensor_a, lambdas)  # x-direction
    dimp_ten_b_init = create_double_impurity(tensor_b, lambdas)  # x-direction
    """

    double_impurity_tensors = []  # for calculating the energy
    for op in (constants.SX, constants.SY, constants.SZ):
        a = create_double_impurity(tensor_a, lambdas, op)
        b = create_double_impurity(tensor_b, lambdas, op)
        # tensor_a = tensor_rotate(tensor_a)
        # tensor_b = tensor_rotate(tensor_b)
        # lambdas = lambdas_rotate(lambdas)
        double_impurity_tensors.append([a, b])

    # dimp_ten_a_mag = copy.deepcopy(double_impurity_tensors[0][0])
    # dimp_ten_b_mag = copy.deepcopy(double_tensor_b)

    # double_impurity_6ring = [None] * 6  # A, B, C, D, E, F - double impurity tensors

    sx, sy, sz, _ = constants.get_spin_operators(spin)

    spin_rotation_operators = None

    if spin == "1/2":
        spin_rotation_operators = (sz, sy, sx)
    if spin == "1":
        spin_rotation_operators = (constants.UZ, constants.UY, constants.UX)
        # spin_rotation_operators = (sz, sy, sx)

    tensors = (tensor_a, tensor_b)

    # for i in range(6):  # impurity_6_ring initialization used for flux calculation
    #    double_impurity_6ring[i] = create_double_impurity(tensors[i % 2], lambdas, spin_rotation_operators[i % 3])

    # operator = (sx / 2, sy / 2, sz / 2)  # operators for Heisenberg model energy calculation
    # operator = operators[0] * 2  # operators for Kitaev model energy calculation

    # impurity_6_ring initialization used for energy calculation
    # double_impurity_6ring[0] = create_double_impurity(tensor_a, lambdas, sx)  # A
    # double_impurity_6ring[1] = create_double_impurity(tensor_b, lambdas, sx)  # B
    # double_impurity_6ring[2] = copy.deepcopy(double_tensor_a)  # C
    # double_impurity_6ring[3] = copy.deepcopy(double_tensor_b)  # D
    # double_impurity_6ring[4] = copy.deepcopy(double_tensor_a)  # E
    # double_impurity_6ring[5] = copy.deepcopy(double_tensor_b)  # F

    # double_impurity_6ring_helper = [None] * 6  # A, B, C, D, E, F - double impurity tensors - helpers
    # for i in range(6):
    #    double_impurity_6ring_helper[i] = create_double_impurity(tensors[i % 2], lambdas, np.eye(d))

    # dimp_ten_a = copy.deepcopy(double_impurity_tensors[0][0])
    # create_double_impurity(tensor_a, lambdas, constants.SX)
    # dimp_ten_b = copy.deepcopy(double_impurity_tensors[0][1])
    # create_double_impurity(tensor_b, lambdas, constants.SX)

    operator = 1j * sx
    dimp_ten_a = create_double_impurity(tensor_a, lambdas, operator)
    dimp_ten_b = create_double_impurity(tensor_b, lambdas, operator)

    tensor_a, tensor_b, lambdas = None, None, None

    # norm = partition_function(*double_impurity_6ring_helper)

    # calculation of the norm
    ten_a = ten_c = ten_e = double_tensor_a
    ten_b = ten_d = ten_f = double_tensor_b
    norm = partition_function(ten_a, ten_b, ten_c, ten_d, ten_e, ten_f)

    ten_a = ten_c = ten_e = double_tensor_a
    ten_b = ten_d = ten_f = double_tensor_b
    ten_a = dimp_ten_a
    ten_b = dimp_ten_b
    measurement = partition_function(ten_a, ten_b, ten_c, ten_d, ten_e, ten_f)

    # measurement1 = partition_function(*double_impurity_6ring)
    print('norm', norm)
    # print('measurement', measurement)
    # print('flux', measurement / norm)
    print('energy', 1.5 * measurement / norm)

    # impurity_plaquette = create_plaquette(*double_impurity_6ring)  # {x xx y yy z zz}
    # measurement2 = np.einsum('x x y y z z->', impurity_plaquette)
    # print('measurement12', measurement1)

    # return measurement1 / norm, 0
    # return measurement2 / norm, 0

    energy = 1
    energy_mem = -1
    num_of_iter = 0

    ################################################################

    energy_six_directions(double_tensor_a, double_tensor_b, double_impurity_tensors, num_of_iter)

    # o = np.tensordot(dimp_ten_a, dimp_ten_b, axes=([0, 1, 2], [0, 1, 2]))
    # o = np.tensordot(double_impurity_tensors[0][0], double_impurity_tensors[0][1], axes=([0, 1, 2], [0, 1, 2]))
    # norm = np.tensordot(double_tensor_a, double_tensor_b, axes=([0, 1, 2], [0, 1, 2]))
    # norm = partition_function(*double_impurity_6ring_helper)
    # print('partition function', norm)

    ################################################################

    num_of_iter += 1

    # for _ in range(100):
    while abs(energy - energy_mem) > 1.E-6:

        # print(double_tensor_a.shape)
        # print(double_tensor_b.shape)
        # print(dimp_ten_a[2].shape)
        # print(dimp_ten_b[2].shape)

        deformed_tensors = create_deformed_tensors(double_tensor_a, double_tensor_b, dim_cut)

        ###########################################

        """
        deformed_12ring = create_deformed_12ring(double_impurity_6ring, double_tensor_a, double_tensor_b, dim_cut)
        double_impurity_6ring = update_6ring(deformed_12ring, deformed_tensors)

        deformed_12ring_helper = create_deformed_12ring(double_impurity_6ring_helper, double_tensor_a, double_tensor_b, dim_cut)
        double_impurity_6ring_helper = update_6ring(deformed_12ring_helper, deformed_tensors)
        """

        ###########################################

        # impurity update is here

        # dimp_sx_a_mag, dimp_sx_b_mag, _ = deform_and_renorm_impurity(dimp_ten_a_mag, dimp_ten_b_mag, dim_cut)

        deformed_imp_tensors = create_deformed_tensors(*zip(*double_impurity_tensors), dim_cut)

        # dimp_ten_a_mag = update_double_tensor(dimp_sx_a_mag, deformed_tensors[1][0], deformed_tensors[2][0])
        # dimp_ten_b_mag = update_double_tensor(dimp_sx_b_mag, deformed_tensors[1][1], deformed_tensors[2][1])

        # dimp_sx_a, dimp_sx_b, _ = \
        # deform_and_renorm_impurity(double_impurity_tensors[0][0], double_impurity_tensors[0][1], dim_cut)

        # dimp_sx_a, dimp_sx_b, _ = deform_and_renorm_impurity(dimp_ten_a, dimp_ten_b, dim_cut)
        # dimp_ten_a = update_double_tensor(dimp_sx_a, deformed_tensors[1][0], deformed_tensors[2][0])
        # dimp_ten_b = update_double_tensor(dimp_sx_b, deformed_tensors[1][1], deformed_tensors[2][1])

        double_impurity_tensors[0][0] = \
            update_double_tensor(deformed_imp_tensors[0][0], deformed_tensors[1][0], deformed_tensors[2][0])

        double_impurity_tensors[0][1] = \
            update_double_tensor(deformed_imp_tensors[0][1], deformed_tensors[1][1], deformed_tensors[2][1])

        double_impurity_tensors[1][0] = \
            update_double_tensor(deformed_tensors[0][0], deformed_imp_tensors[1][0], deformed_tensors[2][0])
        double_impurity_tensors[1][1] = \
            update_double_tensor(deformed_tensors[0][1], deformed_imp_tensors[1][1], deformed_tensors[2][1])

        double_impurity_tensors[2][0] = \
            update_double_tensor(deformed_tensors[0][0], deformed_tensors[1][0], deformed_imp_tensors[2][0])
        double_impurity_tensors[2][1] = \
            update_double_tensor(deformed_tensors[0][1], deformed_tensors[1][1], deformed_imp_tensors[2][1])

        ###########################################

        double_tensor_a = update_double_tensor(*(x[0] for x in deformed_tensors))
        norm_a = np.max(np.abs(double_tensor_a))
        double_tensor_b = update_double_tensor(*(x[1] for x in deformed_tensors))
        norm_b = np.max(np.abs(double_tensor_b))

        # norm_a = norm_b = max(norm_a, norm_b)

        print('norm_a', norm_a)
        print('norm_b', norm_b)

        ###########################################

        double_tensor_a /= norm_a
        double_tensor_b /= norm_b

        ###########################################

        # rot_y = rot_neg_z = lambda ten3: np.transpose(ten3, (1, 2, 0))  # rotate by 1
        # tensor_norms = (norm_a, norm_b)

        """
        for position in range(6):
            double_impurity_6ring[position] /= tensor_norms[position % 2]
            double_impurity_6ring[position] = rot_y(double_impurity_6ring[position])

            double_impurity_6ring_helper[position] /= tensor_norms[position % 2]
            double_impurity_6ring_helper[position] = rot_y(double_impurity_6ring_helper[position])
        """

        ###########################################

        double_impurity_tensors[0][0] /= norm_a
        double_impurity_tensors[0][1] /= norm_b
        double_impurity_tensors[1][0] /= norm_a
        double_impurity_tensors[1][1] /= norm_b
        double_impurity_tensors[2][0] /= norm_a
        double_impurity_tensors[2][1] /= norm_b

        ###########################################

        # dimp_ten_a /= norm_a
        # dimp_ten_b /= norm_b

        # dimp_ten_a_mag /= norm_a
        # dimp_ten_b_mag /= norm_b

        ###########################################

        # energy calculation

        # o = np.tensordot(dimp_ten_a, dimp_ten_b, axes=([0, 1, 2], [0, 1, 2]))
        # o = np.tensordot(double_impurity_tensors[0][0], double_impurity_tensors[0][1], axes=([0, 1, 2], [0, 1, 2]))
        # norm = np.tensordot(double_tensor_a, double_tensor_b, axes=([0, 1, 2], [0, 1, 2]))

        energy_list = energy_six_directions(double_tensor_a, double_tensor_b, double_impurity_tensors, num_of_iter)
        ox1, ox2, oy1, oy2, oz1, oz2 = energy_list

        # print('Expect. iter:', num_of_iter, 'energy:', - (O / norm) / 4)

        # energy_6ring = partition_function(*double_impurity_6ring) / norm
        # print('energy_6ring', - 3 * energy_6ring / 2)
        # print('flux', energy_6ring)

        energy_mem = energy
        # energy = energy_6ring
        energy = ox1
        # energy = (ox1 + ox2 + oy1 + oy2 + oz1 + oz2) / (6 * norm)
        # energy = O / norm
        print('energy', 3 * energy / 2)

        # Magnetization calculation at position 0
        """
        ten_a = ten_c = ten_e = double_tensor_a
        ten_b = ten_d = ten_f = double_tensor_b
        ten_a = dimp_ten_a_mag
        ten_b = dimp_ten_b_mag
        sx_0 = partition_function(ten_a, ten_b, ten_c, ten_d, ten_e, ten_f)
        """
        # print('<S>', sx_0 / norm)

        num_of_iter += 1

    # return energy, sx_0 / norm, num_of_iter
    return energy, num_of_iter
