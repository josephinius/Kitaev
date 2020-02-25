import math
import copy
import numpy as np
from scipy import linalg
import constants
import honeycomb_expectation
# import time
# import pickle
from tqdm import tqdm


EPS = constants.EPS


def construct_kitaev_hamiltonian(h, spin, k=1.):
    """Returns list of two-site Hamiltonian in [x, y, z]-direction for Kitaev model"""
    sx, sy, sz, one = constants.get_spin_operators(spin)
    hamiltonian = - k * np.array([np.kron(sx, sx), np.kron(sy, sy), np.kron(sz, sz)])
    hamiltonian -= h * (np.kron(sx, one) + np.kron(one, sx) +
                        np.kron(sy, one) + np.kron(one, sy) +
                        np.kron(sz, one) + np.kron(one, sz)) / (3 * math.sqrt(3))
    return hamiltonian


def construct_heisenberg_hamiltonian(h, spin, k=1.):
    """
    Returns list of two-site Hamiltonian in [x, y, z]-direction for Heisenberg model.

    Notice that the same Hamiltonian term is used in all three directions
    in order to achieve compatibility with the rest of the code.

    """
    sx, sy, sz, one = constants.get_spin_operators(spin)
    hamiltonian = k * (np.kron(sx, sx) + np.kron(sy, sy) + np.kron(sz, sz))
    hamiltonian += h * (np.kron(sz, one) + np.kron(one, sz)) / 3
    return np.array([hamiltonian, hamiltonian, hamiltonian]) / 2


def construct_ising_hamiltonian(h, spin, k=1.):
    """
    Returns list of two-site Hamiltonian in [x, y, z]-direction for transverse-field Ising model.

    Notice that the same Hamiltonian term is used in all three directions
    in order to achieve compatibility with the rest of the code.

    """
    sx, sy, sz, one = constants.get_spin_operators(spin)
    hamiltonian = - k * np.kron(sx, sx) - h * (np.kron(sz, one) + np.kron(one, sz)) / 2
    return np.array([hamiltonian, hamiltonian, hamiltonian]) / 2


def construct_ite_operator(tau, hamiltonian):
    """Returns imaginary-time evolution (ITE) operator."""
    # return constants.exponentiation(- tau, hamiltonian)  # It seems that this is numerically less accurate.
    return linalg.expm(- tau * hamiltonian)


def apply_gas_operator(tensor, gas_operator):
    """Returns local tensor with loop/string gas operator attached."""
    d, dx, dy, dz = tensor.shape
    tensor = np.einsum('s t i j k, t l m n->s i l j m k n', gas_operator, tensor)
    return tensor.reshape((d, 2 * dx, 2 * dy, 2 * dz))


def calculate_norm(lam):
    norm = np.tensordot(lam, lam, axes=(0, 0))
    return math.sqrt(norm)


def calculate_tensor_norm(ten):
    ten = ten.reshape(-1)
    return np.tensordot(ten, np.conj(ten), axes=(0, 0))


def save_array(np_array, file_name):
    np.save(file_name, np_array)


def tensor_rotate(ten):
    return np.transpose(ten, (0, 2, 3, 1))


def lambdas_rotate(lam):
    return lam[1:] + [lam[0]]


def equal_list_eps(list1, list2, eps=1.E-10):
    for a, b in zip(list1, list2):
        if abs(a - b) > eps:
            return False
    return True


def array_difference(a1, a2):
    l = min(len(a1), len(a2))
    return np.sum(np.abs(a1[:l] - a2[:l]))


def pair_contraction(ten_a, ten_b, lambdas):

    """
    Local tensors: ten_a, ten_b
    Physical bonds (vertical legs): (i), (j)
    Virtual bonds (horizontal legs): x, y1, z1, y2, z2

           z1  (i)       y2
            \  /         /
             \/____x____/  ten_b
      ten_a  /         /\
            /         /  \
           y1       (j)  z2

    """

    # ten_a = ten_a * np.sqrt(lambdas[0])[None, :, None, None]
    ten_a = ten_a * lambdas[0][None, :, None, None]
    ten_a = ten_a * lambdas[1][None, None, :, None]
    ten_a = ten_a * lambdas[2][None, None, None, :]

    # ten_b = ten_b * np.sqrt(lambdas[0])[None, :, None, None]
    ten_b = ten_b * lambdas[1][None, None, :, None]
    ten_b = ten_b * lambdas[2][None, None, None, :]

    pair = np.tensordot(ten_a, ten_b, axes=(1, 1))  # pair_{i y1 z1 j y2 z2} = ten_a{i x y1 z1} * ten_b{j x y2 z2}

    return pair


def apply_gate(u_gate, pair):
    """Returns a product of two-site gate and tensor pair."""
    # theta_{i j y1 z1 y2 z2} = u_gate_{i j k l} * pair_{k y1 z1 l y2 z2}
    theta = np.tensordot(u_gate, pair, axes=([2, 3], [0, 3]))
    return np.transpose(theta, (0, 2, 3, 1, 4, 5))  # theta_{i y1 z1 j y2 z2}


def tensor_pair_update(ten_a, ten_b, theta, lambdas, normalize=True):
    da, db = ten_a.shape, ten_b.shape
    # print('da', da)
    # print('db', db)
    theta = theta.reshape((d * da[2] * da[3], d * db[2] * db[3]))  # theta_{(i y1 z1), (j y2 z2)}

    # theta /= np.max(np.abs(theta))

    x, ss, y = linalg.svd(theta, lapack_driver='gesvd')
    # x, ss, y = linalg.svd(theta, lapack_driver='gesdd')

    # print('ss', ss)
    # norm = ss[0]
    if normalize:
        ss = ss / sum(ss)

    dim_new = min(ss.shape[0], D)

    lambda_new = []
    for i, s in enumerate(ss[:dim_new]):
        if s < EPS:
            print(f'In the ITE procedure: truncating singular values due to small value at index {i}')
            print('Singular values', ss[:dim_new])
            break
        lambda_new.append(s)

    dim_new = len(lambda_new)

    lambda_new = np.array(lambda_new)
    # lambda_new = lambda_new / calculate_norm(lambda_new)
    # norm = np.max(lambda_new)

    # norm = lambda_new[0]
    # lambda_new = lambda_new / norm
    # lambda_new = lambda_new / sum(lambda_new)
    # print('norm in tensor_pair_update', norm)

    x = x[:, :dim_new]
    y = y[:dim_new, :]

    x = x.reshape((d, da[2], da[3], dim_new))
    x = x.transpose((0, 3, 1, 2))

    y = y.reshape((dim_new, d, db[2], db[3]))
    y = y.transpose((1, 0, 2, 3))

    x /= lambdas[1][None, None, :, None]
    x /= lambdas[2][None, None, None, :]

    y /= lambdas[1][None, None, :, None]
    y /= lambdas[2][None, None, None, :]

    return x, y, lambda_new


def update_step(ten_a, ten_b, lambdas, u_gates=None):

    # print('in update_step')

    for i in range(3):

        # print(f'step {i}')

        pair = pair_contraction(ten_a, ten_b, lambdas)

        if u_gates is not None:
            theta = apply_gate(u_gates[i], pair)
        else:
            theta = pair

        normalize = u_gates is not None
        ten_a, ten_b, lambdas[0] = tensor_pair_update(ten_a, ten_b, theta, lambdas, normalize)

        # print('lam updated', lambdas)

        # print('ten_a.shape', ten_a.shape)
        # print('ten_b.shape', ten_b.shape)

        ten_a = tensor_rotate(ten_a)
        ten_b = tensor_rotate(ten_b)
        lambdas = lambdas_rotate(lambdas)

        # print('lam rotated', lambdas)

    # norm_a = np.max(np.abs(ten_a))
    # norm_b = np.max(np.abs(ten_b))
    # ten_a /= norm_a
    # ten_b /= norm_b
    # print('norms in update', norm_a, norm_b)
    # lambdas[0] = lambdas[0] / sum(lambdas[0])
    # lambdas[1] = lambdas[1] / sum(lambdas[1])
    # lambdas[2] = lambdas[2] / sum(lambdas[2])

    return ten_a, ten_b, lambdas


def psi_sigma_psi(psi, sigma):
    temp = np.tensordot(sigma, psi.reshape(-1), axes=(1, 0))
    return np.tensordot(np.conj(psi.reshape(-1)), temp, axes=(0, 0))


def psi_zero_test(psi, spin):
    sx, sy, sz, _ = constants.get_spin_operators(spin)
    sigmas = sx, sy, sz
    # print('psi_sigma_psi(psi, s)', psi_sigma_psi(psi, constants.SX))
    return sum(abs(psi_sigma_psi(psi, s) - 1 / math.sqrt(3)) for s in sigmas)


def prepare_magnetized_state(tensor_a, tensor_b, spin):
    """Returns state |0> = (1 1 1)"""

    print('In prepare_magnetized_state...')

    tau = 100.
    sx, sy, sz, _ = constants.get_spin_operators(spin)
    # op = construct_ite_operator(tau, - constants.SX - constants.SY - constants.SZ)
    op = construct_ite_operator(- tau, sx + sy + sz)

    i = 0
    while psi_zero_test(tensor_a, spin) > 1.E-15:
        tensor_a = np.tensordot(op, tensor_a, axes=(1, 0))
        tensor_a = tensor_a / math.sqrt(np.real(calculate_tensor_norm(tensor_a)))
        if i % 10 == 0:
            print(i, psi_zero_test(tensor_a, spin))
            # tau /= 1.0001
            # op = construct_ite_operator(tau, - constants.SX - constants.SY - constants.SZ)
        i += 1

    print(i, psi_zero_test(tensor_a, spin))

    # tau = 10.

    i = 0
    while psi_zero_test(tensor_b, spin) > 1.E-15:
        tensor_b = np.tensordot(op, tensor_b, axes=(1, 0))
        tensor_b = tensor_b / math.sqrt(np.real(calculate_tensor_norm(tensor_b)))
        if i % 10 == 0:
            print(i, psi_zero_test(tensor_b, spin))
            # tau *= 1.0001
            # op = construct_ite_operator(tau, - constants.SX - constants.SY - constants.SZ)
        i += 1

    print(i, psi_zero_test(tensor_b, spin))

    return tensor_a, tensor_b


def kitaev_spin_one_half_ite_operator(tau):
    # TODO: implement
    pass


def kitaev_spin_one_ite_operator(tau):
    """Returns imaginary time evolution operators for spin=1 Kitaev model exponentiated analytically"""

    # u_gate_x = linalg.expm(- tau * two_site_hamiltonian_x)
    # u_gate_y = linalg.expm(- tau * two_site_hamiltonian_y)
    # u_gate_z = linalg.expm(- tau * two_site_hamiltonian_z)

    a = math.exp(-tau) * (6 * math.exp(tau) + math.exp(2 * tau) + 1.) / 8  # 1/8 e^(-τ) (6 e^τ + e^(2 τ) + 1)
    b = math.exp(-tau) * math.pow((math.exp(tau) - 1.), 2.) / 8  # 1/8 e^(-τ) (e^τ - 1)^2
    c = math.exp(-tau) * (math.exp(2 * tau) - 1.) / 4  # 1/4 e^(-τ) (e^(2 τ) - 1)
    e = math.exp(-tau) * math.pow((math.exp(tau) + 1.), 2.) / 4  # 1/4 e^(-τ) (e^τ + 1)^2
    f = math.exp(-tau) * (math.exp(2 * tau) + 1.) / 2  # 1/2 e^(-τ) (e^(2 τ) + 1)

    u_gate_x = np.array(
        [
            [a, 0, b, 0, c, 0, b, 0, b],
            [0, e, 0, c, 0, c, 0, 2 * b, 0],
            [b, 0, a, 0, c, 0, b, 0, b],
            [0, c, 0, e, 0, 2 * b, 0, c, 0],
            [c, 0, c, 0, f, 0, c, 0, c],
            [0, c, 0, 2 * b, 0, e, 0, c, 0],
            [b, 0, b, 0, c, 0, a, 0, b],
            [0, 2 * b, 0, c, 0, c, 0, e, 0],
            [b, 0, b, 0, c, 0, b, 0, a],
        ], dtype=complex
    )

    u_gate_y = np.array(
        [
            [ a,      0, -b,      0, -c,      0, -b,      0,  b],
            [ 0,      e,  0,      c,  0,     -c,  0, -2 * b,  0],
            [-b,      0,  a,      0,  c,      0,  b,      0, -b],
            [ 0,      c,  0,      e,  0, -2 * b,  0,     -c,  0],
            [-c,      0,  c,      0,  f,      0,  c,      0, -c],
            [ 0,     -c,  0, -2 * b,  0,      e,  0,      c,  0],
            [-b,      0,  b,      0,  c,      0,  a,      0, -b],
            [ 0, -2 * b,  0,     -c,  0,      c,  0,      e,  0],
            [ b,      0, -b,      0, -c,      0, -b,      0,  a],
        ], dtype=complex
    )

    u_gate_z = np.eye(9, dtype=complex)
    u_gate_z[0][0] = math.exp(tau)
    u_gate_z[2][2] = math.exp(-tau)
    u_gate_z[6][6] = math.exp(-tau)
    u_gate_z[8][8] = math.exp(tau)

    return u_gate_x, u_gate_y, u_gate_z


def dimer_gas_operator(phi):
    """Returns dimer gas operator for spin=1 Kitaev model"""
    spin = "1"
    zeta = np.zeros((2, 2, 2), dtype=complex)  # tau_tensor_{i j k}
    zeta[0][0][0] = math.cos(phi)
    zeta[1][0][0] = zeta[0][1][0] = zeta[0][0][1] = math.sin(phi)
    # zeta[0][1][1] = zeta[1][0][1] = zeta[1][1][0] = math.sin(phi)
    sx, sy, sz, one = constants.get_spin_operators(spin)
    d = one.shape[0]
    R = np.zeros((d, d, 2, 2, 2), dtype=complex)  # Q_LG_{s s' i j k}
    for i in range(2):
        for j in range(2):
            for k in range(2):
                temp = np.eye(d)
                if i == 0:
                    temp = temp @ sx  # constants.UX
                if j == 0:
                    temp = temp @ sy  # constants.UY
                if k == 0:
                    temp = temp @ sz  # constants.UZ
                for s in range(d):
                    for sp in range(d):
                        R[s][sp][i][j][k] = zeta[i][j][k] * temp[s][sp]
    return R


########################################################################################################################

# TODO: option for Heisenberg model

model = "Kitaev"
# model = "Heisenberg"

spin = "1"  # implemented options so far: spin = "1", "1/2"

k = 1.
h = 0.E-14
# print('field', h)
D = 4

########################################################################################################################

d = None  # physical dimension

if spin == "1/2":
    d = 2
elif spin == "1":
    d = 3
else:
    raise ValueError('spin should be either "1" or "1/2" (specified as string type)')

if model == "Kitaev":
    construct_hamiltonian = construct_kitaev_hamiltonian
elif model == "Heisenberg":
    construct_hamiltonian = construct_heisenberg_hamiltonian
else:
    raise ValueError('model should be either "Kitaev" or "Heisenberg"')

xi = 1  # initial virtual (bond) dimension
# D = 1  # max virtual (bond) dimension

# tensor_a = np.ones((d, xi, xi, xi)) / math.sqrt(d)
# tensor_b = np.ones((d, xi, xi, xi)) / math.sqrt(d)

# tensor_a = np.ones((d, xi, xi, xi), dtype=complex)
# tensor_b = np.ones((d, xi, xi, xi), dtype=complex)

tensor_a = np.zeros((d, xi, xi, xi), dtype=complex)
tensor_b = np.zeros((d, xi, xi, xi), dtype=complex)

# Spin=1 Kitaev model polarized state:

tensor_a[0][0][0][0] = - 1j * (2 + math.sqrt(3))
tensor_a[1][0][0][0] = (1 - 1j) * (math.sqrt(2) + math.sqrt(6)) / 2
tensor_a[2][0][0][0] = 1

tensor_b[0][0][0][0] = - 1j * (2 + math.sqrt(3))
tensor_b[1][0][0][0] = (1 - 1j) * (math.sqrt(2) + math.sqrt(6)) / 2
tensor_b[2][0][0][0] = 1

"""
tensor_a = np.zeros((d, xi, xi, xi))
tensor_b = np.zeros((d, xi, xi, xi))
tensor_a[0][0][0][0] = 1.
tensor_a[1][0][0][0] = 1.
tensor_b[0][0][0][0] = 1.
"""

# norm_a = np.max(np.abs(tensor_a))
# norm_b = np.max(np.abs(tensor_b))
# tensor_a /= norm_a
# tensor_b /= norm_b

# tensor_a = tensor_a / math.sqrt(np.real(calculate_tensor_norm(tensor_a)))
# tensor_b = tensor_b / math.sqrt(np.real(calculate_tensor_norm(tensor_b)))

# print('polarization test', psi_zero_test(tensor_a, spin))

lambdas = [np.array([1., ], dtype=complex), np.array([1., ], dtype=complex), np.array([1., ], dtype=complex)]

# tensor_a, tensor_b = prepare_magnetized_state(tensor_a, tensor_b, spin)
# print('magnetized state')
# print(tensor_a)
# print(tensor_b)

Q_LG = constants.create_loop_gas_operator(spin)

# print(Q_LG)
# print(tensor_a.shape)

# tensor_a = np.einsum('s t i j k, t l m n->s i l j m k n', constants.Q_LG, tensor_a)
# tensor_a = tensor_a.reshape((d, 2, 2, 2))
tensor_a = apply_gas_operator(tensor_a, Q_LG)

# tensor_b = np.einsum('s t i j k, t l m n->s i l j m k n', constants.Q_LG, tensor_b)
# tensor_b = tensor_b.reshape((d, 2, 2, 2))
tensor_b = apply_gas_operator(tensor_b, Q_LG)

# tensor_a = tensor_a / math.sqrt(np.real(calculate_tensor_norm(tensor_a)))
# tensor_b = tensor_b / math.sqrt(np.real(calculate_tensor_norm(tensor_b)))

"""
# String-Gas state
phi = math.pi * 0.275
R = dimer_gas_operator(phi)
tensor_a = apply_gas_operator(tensor_a, R)
tensor_b = apply_gas_operator(tensor_b, R)
"""

# lambdas = [np.array([1., 1.]) / math.sqrt(2), np.array([1., 1.]) / math.sqrt(2), np.array([1., 1.]) / math.sqrt(2)]
lambdas = [np.array([1., 1.], dtype=complex), np.array([1., 1.], dtype=complex), np.array([1., 1.], dtype=complex)]
# lambdas = [np.ones((4,), dtype=complex) / 2, np.ones((4,), dtype=complex) / 2, np.ones((4,), dtype=complex) / 2]
# lambdas = [np.ones((4,), dtype=complex), np.ones((4,), dtype=complex), np.ones((4,), dtype=complex)]

# tensor_a = tensor_a / math.sqrt(np.real(calculate_tensor_norm(tensor_a)))
# tensor_b = tensor_b / math.sqrt(np.real(calculate_tensor_norm(tensor_b)))

# print(tensor_a.shape)

# tau_initial = 4.E-3
tau_initial = 1.E-2
tau_final = 1.E-6
# u_gates = [u_gate_x, u_gate_y, u_gate_z]
# u_gates = np.array([construct_ITE_operator(tau, hamiltonian).reshape(d, d, d, d) for hamiltonian in H])

tau = tau_initial
# tau = tau_final

refresh = 100

file_name = 'kitaev.txt'  # output file

energy = 1

"""
energy, mag_x, num_of_iter = honeycomb_expectation.kitaevS1_GS_expectation_calculation(tensor_a, tensor_b, lambdas, D)
energy = - 3 * energy / 2

print('Energy of the initial state', energy, 'mag_x:', mag_x, 'num_of_iter', num_of_iter)
"""

energy, num_of_iter = honeycomb_expectation.coarse_graining_procedure(tensor_a, tensor_b, lambdas, D)
print('Energy of the initial state', 3 * energy / 2, 'num_of_iter', num_of_iter)
# print('Flux of the initial state', energy, 'num_of_iter', num_of_iter)

with open(file_name, 'w') as f:
    f.write('# Kitaev S=%s model - ITE flow\n' % spin)
    f.write('# D=%d, tau=%.8E, h=%.14E\n' % (D, tau, h))
    f.write('# Iter\t\tEnergy\t\t\tCoarse-grain steps\n')
f = open(file_name, 'a')
# f.write('%d\t\t%.15f\t%.15f\t%d\n' % (0, np.real(energy), np.real(mag_x), num_of_iter))
f.write('%d\t\t%.15f\t%d\n' % (0, 3 * np.real(energy) / 2, num_of_iter))
# f.write('%d\t\t%.15f\t%d\n' % (0, np.real(energy), num_of_iter))
f.close()

energy_old = -1

lambdas_memory = copy.deepcopy(lambdas)

# H = construct_hamiltonian(h, spin, k)
# u_gates = np.array([construct_ite_operator(tau, hamiltonian).reshape(d, d, d, d) for hamiltonian in H])
u_gates = [exp_ham.reshape(d, d, d, d) for exp_ham in kitaev_spin_one_ite_operator(tau)]

j = 0  # ITE-step index

while tau >= tau_final and (j * refresh < 10100):

    for i in tqdm(range(refresh)):
        tensor_a, tensor_b, lambdas = update_step(tensor_a, tensor_b, lambdas, u_gates)

    print('iter', (j + 1) * refresh)
    print('tau', tau)
    print(lambdas[0][:12])
    print(lambdas[1][:12])
    print(lambdas[2][:12])

    tensor_a_copy = copy.deepcopy(tensor_a)
    tensor_b_copy = copy.deepcopy(tensor_b)
    lambdas_copy = copy.deepcopy(lambdas)

    for i in range(1):
        tensor_a_copy, tensor_b_copy, lambdas_copy = update_step(tensor_a_copy, tensor_b_copy, lambdas_copy)
        tensor_a_copy = tensor_a_copy / np.max(np.abs(tensor_a_copy))
        tensor_b_copy = tensor_b_copy / np.max(np.abs(tensor_b_copy))
        lambdas_copy = [lam / lam[0] for lam in lambdas_copy]

    print(lambdas_copy[0][:12])
    print(lambdas_copy[1][:12])
    print(lambdas_copy[2][:12])

    energy, num_of_iter = honeycomb_expectation.coarse_graining_procedure(tensor_a_copy, tensor_b_copy, lambdas_copy, D)
    # energy, num_of_iter = honeycomb_expectation.coarse_graining_procedure(tensor_a, tensor_b, lambdas, D)

    energy = 3 * energy / 2
    # energy = energy / 4

    # print('# ITE flow iter:', (j + 1) * refresh, 'energy:', energy, 'mag_x:', mag_x, 'num_of_iter:', num_of_iter)
    print('# ITE flow iter:', (j + 1) * refresh, 'energy:', energy, 'num_of_iter:', num_of_iter)

    f = open(file_name, 'a')
    # f.write('%d\t\t%.15f\t%.15f\t%d\n' % ((j + 1) * refresh, np.real(energy), np.real(mag_x), num_of_iter))
    f.write('%d\t\t%.15f\t%.15f\t%d\n' % ((j + 1) * refresh, np.real(energy), tau, num_of_iter))
    f.close()

    # test1 = equal_list_eps(lambdas[0], lambdas_memory[0])
    s1 = array_difference(lambdas[0], lambdas_memory[0])
    # test2 = equal_list_eps(lambdas[1], lambdas_memory[1])
    s2 = array_difference(lambdas[1], lambdas_memory[1])
    # test3 = equal_list_eps(lambdas[2], lambdas_memory[2])
    s3 = array_difference(lambdas[2], lambdas_memory[2])
    lambdas_memory = copy.deepcopy(lambdas)
    print(s1 + s2 + s3)
    # if s1 < 1.E-11 and s2 < 1.E-11 and s3 < 1.E-11:
    if s1 < 1.E-6 and s2 < 1.E-6 and s3 < 1.E-6:
        print('lambdas converged')
        # print('decreasing tau')
        # tau /= 3
        # tau /= 10
        # u_gates = np.array([construct_ite_operator(tau, hamiltonian).reshape(d, d, d, d) for hamiltonian in H])
        # u_gates = [exp_ham.reshape(d, d, d, d) for exp_ham in kitaev_spin_one_ite_operator(tau)]
    j += 1

"""
while abs(energy - energy_old) >= 1.E-10 and (j * refresh < 2000):

    for i in tqdm(range(refresh)):
        # print('i', i)
        tensor_a, tensor_b, lambdas = update_step(tensor_a, tensor_b, lambdas, u_gates)
        # if i % 1 == 0:
        #    print('#', lambdas[0][:D])
        #    print('#', lambdas[1][:D])
        #    print('#', lambdas[2][:D])

    print('iter', (j + 1) * refresh)
    # print('#', lambdas[0][:D])
    # print('#', lambdas[1][:D])
    # print('#', lambdas[2][:D])

    if (j + 1) * refresh >= 0:
        energy_mem = energy
        # energy, mag_x, num_of_iter = \
        # honeycomb_expectation.kitaevS1_GS_expectation_calculation(tensor_a, tensor_b, lambdas, D)
        energy, num_of_iter = honeycomb_expectation.kitaev_expectation_calculation(tensor_a, tensor_b, lambdas, D)
        energy = - 3 * energy / 2
        # energy = energy
        # energy = - energy / 4

        # print('# ITE flow iter:', (j + 1) * refresh, 'energy:', energy, 'mag_x:', mag_x, 'num_of_iter:', num_of_iter)
        print('# ITE flow iter:', (j + 1) * refresh, 'energy:', energy, 'num_of_iter:', num_of_iter)

        f = open(file_name, 'a')
        # f.write('%d\t\t%.15f\t%.15f\t%d\n' % ((j + 1) * refresh, np.real(energy), np.real(mag_x), num_of_iter))
        f.write('%d\t\t%.15f\t%d\n' % ((j + 1) * refresh, np.real(energy), num_of_iter))
        f.close()

    j += 1
"""

# TODO: check only the convergence of lambdas (maybe TRG is too inaccurate)
