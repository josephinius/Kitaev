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


def create_star_lattice_tensors(theta):
    """
    Returns Loop Gas (LG) state for KMS (parametrized by variational parameter theta).
    See Eq. (2) and (3) in Ref. [1].
    """

    xi = 1  # initial virtual (bond) dimension

    # tensor_x = np.zeros((d, xi, xi, xi), dtype=complex)
    # tensor_y = np.zeros((d, xi, xi, xi), dtype=complex)
    # tensor_z = np.zeros((d, xi, xi, xi), dtype=complex)

    c = math.cos(theta)
    s = math.sin(theta) * math.sqrt(2) / 2

    _, v = linalg.eigh(-(sx * c + sy * s + sz * s))  # Eq. (2)
    tensor_x = v[:, 0].reshape((d, xi, xi, xi))
    _, v = linalg.eigh(-(sx * s + sy * c + sz * s))  # Eq. (2)
    tensor_y = v[:, 0].reshape((d, xi, xi, xi))
    _, v = linalg.eigh(-(sx * s + sy * s + sz * c))  # Eq. (2)
    tensor_z = v[:, 0].reshape((d, xi, xi, xi))

    # apply loop gas operator on initial state
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

    J = math.tan(phi)

    tensors = (tensor_x, tensor_y, tensor_z)
    ops = (sx, sy, sz)

    create_double_impurity = lambda ten, op: honeycomb_expectation.create_double_impurity(ten, lambdas, op)
    imps = Impurities(*(Impurities(*(create_double_impurity(ten, op) for op in ops)) for ten in tensors))

    # double_x, double_y, double_z
    # imps.a.b

    term1 = create_triangle_pair(x1=double_x, y1=imps.y.x, z1=imps.z.x, x2=double_x, y2=double_y, z2=double_z)
    term2 = create_triangle_pair(x1=imps.x.y, y1=double_y, z1=imps.z.y, x2=double_x, y2=double_y, z2=double_z)
    term3 = create_triangle_pair(x1=imps.x.z, y1=imps.y.z, z1=double_z, x2=double_x, y2=double_y, z2=double_z)

    # term4 = create_triangle_pair(x1=double_x, y1=double_y, z1=double_z, x2=double_x, y2=imps.y.x, z2=imps.z.x)
    # term5 = create_triangle_pair(x1=double_x, y1=double_y, z1=double_z, x2=imps.x.y, y2=double_y, z2=imps.z.y)
    # term6 = create_triangle_pair(x1=double_x, y1=double_y, z1=double_z, x2=imps.x.z, y2=imps.y.z, z2=double_z)

    term7 = create_triangle_pair(x1=imps.x.x, y1=double_y, z1=double_z, x2=imps.x.x, y2=double_y, z2=double_z)

    # w_imp = term7 + J * (term1 + term2 + term3 + term4 + term5 + term6)
    w_imp = term7 + J * 2 * (term1 + term2 + term3)

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

    spin = '1/2'

    d = constants.spin_to_physical_dimension(spin)
    sx, sy, sz, _ = constants.get_spin_operators(spin)

    lambdas = [np.array([1., 1.], dtype=complex),
               np.array([1., 1.], dtype=complex),
               np.array([1., 1.], dtype=complex)]

    dim = 200  # bond dimension
    min_energy_theta = []

    for p in np.linspace(0.48, 0.49, num=1, endpoint=True):
        phi = p * math.pi
        minimum = None

        for t in np.linspace(0.25, 0.45, num=10, endpoint=False):
            theta = t * math.pi
            tensor_x, tensor_y, tensor_z = create_star_lattice_tensors(theta)
            create_double_tensor = lambda x: honeycomb_expectation.create_double_tensor(x, lambdas)
            double_x, double_y, double_z = map(create_double_tensor, (tensor_x, tensor_y, tensor_z))

            triangle = ncon(
                [double_x, double_y, double_z],
                [[-1, 2, 3], [1, -2, 3], [1, 2, -3]]
            )
            w = np.tensordot(triangle, triangle, axes=(0, 0))
            xi = 2
            corners = create_corners(w, xi)
            transfer_matrices = create_tms(w, xi)

            w_imp = create_energy_impurity(phi, tensor_x, tensor_y, tensor_z, double_x, double_y, double_z)

            ctm = ctmrg.CTMRG(dim, weight=w, corners=corners, tms=transfer_matrices, weight_imp=w_imp)
            energy, delta, _, num_of_iter = ctm.ctmrg_iteration()

            if minimum is None or minimum[0] > energy:
                minimum = (energy, t)

        min_energy_theta.append([p, minimum])

    print(min_energy_theta)
