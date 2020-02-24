import math
import numpy as np
from scipy import linalg

EPS = 1.E-32


# Spin=1 operators: SX, SY, SZ

SX = np.array([
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0]
], dtype=complex) / math.sqrt(2)

SY = np.array([
    [0, 1, 0],
    [-1, 0, 1],
    [0, -1, 0]
], dtype=complex) / (math.sqrt(2) * 1j)

SZ = np.array([
    [1., 0, 0],
    [0, 0, 0],
    [0, 0, -1.]
], dtype=complex)


# UX = linalg.expm(1j * math.pi * SX)
# UX = exponentiation(1j * math.pi, SX)
UX = np.array([
    [0, 0, -1.],
    [0, -1., 0],
    [-1., 0, 0]
], dtype=complex)
# print('UX')
# print(UX)

# UY = linalg.expm(1j * math.pi * SY)
# UY = exponentiation(1j * math.pi, SY)
UY = np.array([
    [0, 0, 1.],
    [0, -1., 0],
    [1., 0, 0]
], dtype=complex)

# UZ = linalg.expm(1j * math.pi * SZ)
# UZ = exponentiation(1j * math.pi, SZ)
UZ = np.array([
    [-1., 0, 0],
    [0, 1., 0],
    [0, 0, -1.]
], dtype=complex)

# ID3 = np.eye(3)

# Spin-1/2 operators: sx, sy, sz

sx = np.array([
    [0, 1],
    [1, 0]
], dtype=complex)

sy = np.array([
    [0, -1j],
    [1j, 0]
], dtype=complex)

sz = np.array([
    [1, 0],
    [0, -1]
], dtype=complex)

# id2 = np.eye(2)


def get_spin_operators(spin):
    if spin == "1/2":
        return sx, sy, sz, np.eye(2)
    elif spin == "1":
        return SX, SY, SZ, np.eye(3)
    raise ValueError('spin parameter should be a string: "1/2" or "1"')


def create_loop_gas_operator(spin):
    """Returns loop gas (LG) operator Q_LG for spin=1/2 or spin=1 Kitaev model."""

    tau_tensor = np.zeros((2, 2, 2), dtype=complex)  # tau_tensor_{i j k}
    # tau_tensor[0][0][0] = - 1j
    tau_tensor[0][0][0] = - 1
    tau_tensor[0][1][1] = tau_tensor[1][0][1] = tau_tensor[1][1][0] = 1

    sx, sy, sz, one = get_spin_operators(spin)
    d = one.shape[0]

    Q_LG = np.zeros((d, d, 2, 2, 2), dtype=complex)  # Q_LG_{s s' i j k}

    u_gamma = None

    if spin == "1/2":
        u_gamma = (sx, sy, sz)
    if spin == "1":
        u_gamma = (UX, UY, UZ)

    for i in range(2):
        for j in range(2):
            for k in range(2):
                temp = np.eye(d)
                if i == 0:
                    temp = temp @ u_gamma[0]
                if j == 0:
                    temp = temp @ u_gamma[1]
                if k == 0:
                    temp = temp @ u_gamma[2]
                for s in range(d):
                    for sp in range(d):
                        Q_LG[s][sp][i][j][k] = tau_tensor[i][j][k] * temp[s][sp]

    return Q_LG


def exponentiation(alpha, s):
    """
    Returns matrix exponentiation exp(alpha * s), where alpha is (complex) coefficient and s is (Hermitian) matrix.

    Note: This method of matrix exponentiation is not numerically accurate and is used for debugging purposes only.
    """
    w, v = linalg.eigh(s)
    # w, v = linalg.eig(s)
    d = s.shape[0]
    assert np.allclose(s @ v - v @ np.diag(w), np.zeros((d, d)), rtol=1e-12, atol=1e-14)
    assert np.allclose(s - v @ np.diag(w) @ linalg.inv(v), np.zeros((d, d)), rtol=1e-12, atol=1e-14)
    assert np.allclose(v @ linalg.inv(v), np.eye(d), rtol=1e-12, atol=1e-14)
    # w = np.where(np.abs(np.imag(w)) < 1.E-14, np.real(w), w)
    # w = np.where(np.abs(np.real(w)) < 1.E-14, np.imag(w), w)
    # print('w', w)
    result = v @ np.exp(alpha * np.diag(w)) @ linalg.inv(v)
    # return v @ np.exp(np.diag(alpha * w)) @ linalg.inv(v)
    # result = np.where(np.abs(result) < 5.E-14, 0, result)
    # result = np.where(np.abs(np.imag(result)) < 1.E-15, np.real(result), result)
    # result = np.where(np.abs(np.real(result)) < 1.E-15, np.imag(result), result)
    return result
