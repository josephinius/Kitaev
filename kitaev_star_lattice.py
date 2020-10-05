"""
Kitaev model on star lattice (KMS)
Usage: kitaev_star_lattice.py

References

[1] Phys. Rev. B 101, 035149 (2020)

"""


import math
import numpy as np
from scipy import linalg
from ncon import ncon
from collections import namedtuple

import constants
import honeycomb_expectation
import ctmrg
from kitaev import apply_gas_operator


def create_magnetic_state(theta):
    xi = 1  # initial virtual (bond) dimension

    c, s = math.cos(theta), math.sin(theta) / math.sqrt(2)
    _, v = linalg.eigh(-(sx * c + sy * s + sz * s))  # Eq. (2)
    tensor_x = v[:, 0].reshape((d, xi, xi, xi))
    _, v = linalg.eigh(-(sx * s + sy * c + sz * s))  # Eq. (2)
    tensor_y = v[:, 0].reshape((d, xi, xi, xi))
    _, v = linalg.eigh(-(sx * s + sy * s + sz * c))  # Eq. (2)
    tensor_z = v[:, 0].reshape((d, xi, xi, xi))
    return tensor_x, tensor_y, tensor_z


def create_star_lattice_LG_state(theta):
    """
    Returns Loop Gas (LG) state for KMS (parametrized by variational parameter theta).
    See Eq. (2) and (3) in Ref. [1].
    """

    # tensor_x = np.zeros((d, xi, xi, xi), dtype=complex)
    # tensor_y = np.zeros((d, xi, xi, xi), dtype=complex)
    # tensor_z = np.zeros((d, xi, xi, xi), dtype=complex)

    tensor_x, tensor_y, tensor_z = create_magnetic_state(theta)

    # apply loop gas operator on initial state
    loop_gas = lambda x: apply_gas_operator(x, constants.create_loop_gas_operator(spin))
    tensor_x, tensor_y, tensor_z = map(loop_gas, (tensor_x, tensor_y, tensor_z))

    return tensor_x, tensor_y, tensor_z


def dimer_gas_operator(spin, alpha, beta, intertriangle_direction):
    """Returns dimer gas operator (or variational ansatz) R for spin=1/2 or spin=1 Kitaev model."""

    zeta = np.zeros((2, 2, 2), dtype=complex)  # tau_tensor_{i j k}

    zeta[0][0][0] = math.cos(beta)

    if intertriangle_direction == 'x':
        zeta[1][0][0] = math.sin(beta) * math.sqrt(math.sin(alpha))
        zeta[0][1][0] = zeta[0][0][1] = math.sin(beta) * math.sqrt(math.cos(alpha))

    if intertriangle_direction == 'y':
        zeta[0][1][0] = math.sin(beta) * math.sqrt(math.sin(alpha))
        zeta[1][0][0] = zeta[0][0][1] = math.sin(beta) * math.sqrt(math.cos(alpha))

    if intertriangle_direction == 'z':
        zeta[0][0][1] = math.sin(beta) * math.sqrt(math.sin(alpha))
        zeta[1][0][0] = zeta[0][1][0] = math.sin(beta) * math.sqrt(math.cos(alpha))

    sx, sy, sz, one = constants.get_spin_operators(spin)
    d = one.shape[0]
    R = np.zeros((d, d, 2, 2, 2), dtype=complex)  # R_DG_{s s' i j k}

    p = 1

    for i in range(2):
        for j in range(2):
            for k in range(2):
                temp = np.eye(d)
                if i == p:
                    temp = temp @ sx
                if j == p:
                    temp = temp @ sy
                if k == p:
                    temp = temp @ sz
                for s in range(d):
                    for sp in range(d):
                        R[s][sp][i][j][k] = zeta[i][j][k] * temp[s][sp]
    return R


def create_star_lattice_SG_state(alpha, beta):
    """
    Returns String Gas (SG) state for KMS (parametrized by variational parameters alpha and beta).
    See Section III. B and Appendix B in Ref. [1].
    """

    tensor, _, _ = create_magnetic_state(theta=math.atan(math.sqrt(2)))

    DG_x, DG_y, DG_z = (dimer_gas_operator(spin, alpha, beta, direction) for direction in ('x', 'y', 'z'))

    # apply string gas operator
    tensor_x, tensor_y, tensor_z = (apply_gas_operator(tensor, dg) for dg in (DG_x, DG_y, DG_z))

    # apply loop gas operator
    loop_gas = lambda x: apply_gas_operator(x, constants.create_loop_gas_operator(spin))
    tensor_x, tensor_y, tensor_z = map(loop_gas, (tensor_x, tensor_y, tensor_z))

    return tensor_x, tensor_y, tensor_z


def create_triangle_pair(x1, y1, z1, x2, y2, z2):
    """
    Returns triangle pair constructed as follows:

           y                           z
            \                         /
             \                       /
             (y1)                  (z2)
             |   \ z            y /   |
             |    \       x      /    |
           x |     (x1)------(x2)     | x
             |    /              \    |
             |   / y            z \   |
            (z1)                   (y2)
            /                        \
           /                          \
          z                            y

    """

    def create_triangle(x, y, z):
        return ncon([x, y, z], [[-1, 2, 3], [1, -2, 3], [1, 2, -3]])

    triangle1 = create_triangle(x1, y1, z1)
    triangle2 = create_triangle(x2, y2, z2)

    return np.tensordot(triangle1, triangle2, axes=(0, 0))


def create_energy_impurity(phi, tensor_x, tensor_y, tensor_z, double_x, double_y, double_z):

    """

    1) Terms (1) - (7)

           y                           z
            \                         /
             \                       /
             (y1)                  (z2)
             |   \ 3            5 /   |
             |    \       7      /    |
           1 |     (x1)------(x2)     | 4
             |    /              \    |
             |   / 2            6 \   |
            (z1)                   (y2)
            /                        \
           /                          \
          z                            y

    2) Note on namedtuple representation of impurity double tensors

    impurities.x: tensor_x "sandwitching" sx (impurities.x.x), sy (impurities.x.y), sz (impurities.x.z)
    impurities.y: tensor_y "sandwitching" sx (impurities.y.x), sy (impurities.y.y), sz (impurities.y.z)
    impurities.z: tensor_z "sandwitching" sx (impurities.z.x), sy (impurities.z.y), sz (impurities.z.z)

    Equivalently, impurities.x.x can be written as impurities[0][0], or for example impurities.y.z is impurities[1][2]
    """

    Impurities = namedtuple('Impurities', ['x', 'y', 'z'])

    tensors = (tensor_x, tensor_y, tensor_z)
    ops = (sx, sy, sz)

    create_double_impurity = lambda ten, op: honeycomb_expectation.create_double_impurity(ten, lambdas, op)
    imps = Impurities(*(Impurities(*(create_double_impurity(ten, op) for op in ops)) for ten in tensors))

    # double_x, double_y, double_z
    # imps.a.b

    term1 = create_triangle_pair(x1=double_x, y1=imps.y.x, z1=imps.z.x, x2=double_x, y2=double_y, z2=double_z)
    term2 = create_triangle_pair(x1=imps.x.y, y1=double_y, z1=imps.z.y, x2=double_x, y2=double_y, z2=double_z)
    term3 = create_triangle_pair(x1=imps.x.z, y1=imps.y.z, z1=double_z, x2=double_x, y2=double_y, z2=double_z)

    term4 = create_triangle_pair(x1=double_x, y1=double_y, z1=double_z, x2=double_x, y2=imps.y.x, z2=imps.z.x)
    term5 = create_triangle_pair(x1=double_x, y1=double_y, z1=double_z, x2=imps.x.y, y2=double_y, z2=imps.z.y)
    term6 = create_triangle_pair(x1=double_x, y1=double_y, z1=double_z, x2=imps.x.z, y2=imps.y.z, z2=double_z)

    term7 = create_triangle_pair(x1=imps.x.x, y1=double_y, z1=double_z, x2=imps.x.x, y2=double_y, z2=double_z)

    w_imp = math.sin(phi) * term7 + math.cos(phi) * (term1 + term2 + term3 + term4 + term5 + term6) / 3

    w_imp *= 3 / 2
    w_imp /= 4
    w_imp *= -1

    return w_imp


def create_corners(w, xi):  # TODO: unify with honeycomb_expectation
    c1 = w.reshape((xi, xi, xi * xi, xi * xi, xi, xi))
    c1 = np.einsum('i i j k l l->j k', c1)
    c2 = w.reshape((xi, xi, xi, xi, xi * xi, xi * xi))
    c2 = np.einsum('i i j j k l->k l', c2)
    c3 = w.reshape((xi * xi, xi, xi, xi, xi, xi * xi))
    c3 = np.einsum('i j j k k l->l i', c3)
    c4 = w.reshape((xi * xi, xi * xi, xi, xi, xi, xi))
    c4 = np.einsum('i j k k l l->i j', c4)
    corners = (c1, c2, c3, c4)
    return corners


def create_tms(w, xi):  # TODO: unify with honeycomb_expectation
    t1 = np.einsum('i i j k l->j k l', w.reshape((xi, xi, xi * xi, xi * xi, xi * xi)))
    t2 = np.einsum('i j j k l->k l i', w.reshape((xi * xi, xi, xi, xi * xi, xi * xi)))
    t3 = np.einsum('i j k k l->l i j', w.reshape((xi * xi, xi * xi, xi, xi, xi * xi)))
    t4 = np.einsum('i j k l l->i j k', w.reshape((xi * xi, xi * xi, xi * xi, xi, xi)))
    transfer_matrices = (t1, t2, t3, t4)
    return transfer_matrices


if __name__ == "__main__":

    print('Kitaev model on star lattice (KMS)')
    file_name = "star.txt"

    spin = '1/2'

    d = constants.spin_to_physical_dimension(spin)
    sx, sy, sz, _ = constants.get_spin_operators(spin)

    """
    lambdas = [np.array([1., 1.], dtype=complex),
               np.array([1., 1.], dtype=complex),
               np.array([1., 1.], dtype=complex)]
    """

    lambdas = [np.ones(4, dtype=complex),
               np.ones(4, dtype=complex),
               np.ones(4, dtype=complex)]

    tensor_x, tensor_y, tensor_z = create_star_lattice_SG_state(alpha=0.1 * math.pi, beta=0.2 * math.pi)

    print(tensor_x.shape)
    print(tensor_y.shape)
    print(tensor_z.shape)

    dim = 100
    phi = 0.1

    create_double_tensor = lambda x: honeycomb_expectation.create_double_tensor(x, lambdas)
    double_x, double_y, double_z = map(create_double_tensor, (tensor_x, tensor_y, tensor_z))
    w = create_triangle_pair(double_x, double_y, double_z, double_x, double_y, double_z)
    xi = 4
    corners = create_corners(w, xi)
    transfer_matrices = create_tms(w, xi)
    w_imp = create_energy_impurity(phi, tensor_x, tensor_y, tensor_z, double_x, double_y, double_z)
    ctm = ctmrg.CTMRG(dim, weight=w, corners=corners, tms=transfer_matrices, weight_imp=w_imp)
    energy, delta, _, num_of_iter = ctm.ctmrg_iteration()


    """
    dim = 100  # bond dimension
    min_energy_theta = []

    with open(file_name, 'w') as f:
        f.write('# phi / pi\t\t\tEnergy\t\t\ttheta*\n')

    for p in np.linspace(0.05, 0.45, num=41, endpoint=True):
        phi = p * math.pi
        minimum = None

        for t in np.linspace(0.01, 0.51, num=100, endpoint=False):
            theta = t * math.pi
            tensor_x, tensor_y, tensor_z = create_star_lattice_LG_state(theta)
            create_double_tensor = lambda x: honeycomb_expectation.create_double_tensor(x, lambdas)
            double_x, double_y, double_z = map(create_double_tensor, (tensor_x, tensor_y, tensor_z))

            w = create_triangle_pair(double_x, double_y, double_z, double_x, double_y, double_z)

            xi = 2
            corners = create_corners(w, xi)
            transfer_matrices = create_tms(w, xi)

            w_imp = create_energy_impurity(phi, tensor_x, tensor_y, tensor_z, double_x, double_y, double_z)

            ctm = ctmrg.CTMRG(dim, weight=w, corners=corners, tms=transfer_matrices, weight_imp=w_imp)
            energy, delta, _, num_of_iter = ctm.ctmrg_iteration()

            if minimum is None or minimum[0] > energy:
                minimum = (energy, t)

        min_energy_theta.append([p, minimum])
        f = open(file_name, 'a')
        f.write('%.15f\t\t%.15f\t%.15f\n' % (p, np.real(minimum[0]), minimum[1]))
        f.close()

    print(min_energy_theta)
    """
