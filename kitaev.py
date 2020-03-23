import math
import copy
import numpy as np
from scipy import linalg
import constants
import honeycomb_expectation
# import time
# import pickle
from tqdm import tqdm
import ctmrg


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
    """Returns L2 norm for 1-dimensional (real) numpy array."""

    norm = np.tensordot(lam, lam, axes=(0, 0))
    return math.sqrt(norm)


def complex_inner_product(ten):
    """Returns complex inner product of flattened tensor."""

    ten = ten.reshape(-1)
    return np.tensordot(ten, np.conj(ten), axes=(0, 0))


def save_array(np_array, file_name):
    """Saves numpy array into file; returns nothing."""

    np.save(file_name, np_array)


def tensor_rotate(ten):
    """Returns tensor with virtual indices rotated anti-clock wise."""

    return np.transpose(ten, (0, 2, 3, 1))


def lambdas_rotate(lam):
    """Returns lambdas rotated anti-clock wise."""

    return lam[1:] + [lam[0]]


def abs_list_difference(a1, a2):
    """Returns absolute difference between two 1-dimensional arrays."""

    if len(a1) > len(a2):
        return abs_list_difference(a2, a1)
    n = len(a1)
    return np.sum(np.abs(a1[:n] - a2[:n])) + np.sum(np.abs(a2[n:]))


def pair_contraction(ten_a, ten_b, lambdas):
    """
    Returns result of contraction of two tensors via x leg.

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
    """Returns new tensor pair and corresponding (by default normalized) singular values."""

    da, db = ten_a.shape, ten_b.shape
    theta = theta.reshape((d * da[2] * da[3], d * db[2] * db[3]))  # theta_{(i y1 z1), (j y2 z2)}

    x, ss, y = linalg.svd(theta, lapack_driver='gesvd')  # lapack_driver='gesdd' or 'gesvd'

    if normalize:
        ss = ss / sum(ss)
    # another way of normalizing singular values would be by dividing by norm = ss[0]

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
    """Performs simple update in all three directions and returns updated tensors and singular values."""

    for i in range(3):
        pair = pair_contraction(ten_a, ten_b, lambdas)
        if u_gates is not None:
            theta = apply_gate(u_gates[i], pair)
        else:
            theta = pair
        ten_a, ten_b, lambdas[0] = tensor_pair_update(ten_a, ten_b, theta, lambdas, normalize=u_gates is not None)
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
    """Returns <psi|sigma|psi>."""

    temp = np.tensordot(sigma, psi.reshape(-1), axes=(1, 0))
    return np.tensordot(np.conj(psi.reshape(-1)), temp, axes=(0, 0))


def psi_zero_test(psi, spin):
    """Returns <psi|(sx + sy + sz)|psi> - 1/sqrt(3).

    Note: For magnetized state, we have <psi|(sx + sy + sz)|psi> = 1/sqrt(3).
    """

    sx, sy, sz, _ = constants.get_spin_operators(spin)
    sigmas = sx, sy, sz
    return sum(abs(psi_sigma_psi(psi, s) - 1 / math.sqrt(3)) for s in sigmas)


def prepare_magnetized_state(tensor, spin):
    """Returns magnetized state (i.e. polarized state) |0> = (1 1 1).

    Note: We use analytical form of magnetized state for spin=1 Kitaev model.
    """

    print('In prepare_magnetized_state')

    step = 100.
    sx, sy, sz, _ = constants.get_spin_operators(spin)
    op = construct_ite_operator(- step, sx + sy + sz)

    i = 0
    while psi_zero_test(tensor, spin) > 1.E-15:
        tensor = np.tensordot(op, tensor, axes=(1, 0))
        tensor = tensor / math.sqrt(np.real(complex_inner_product(tensor)))
        if i % 10 == 0:
            print(i, psi_zero_test(tensor, spin))
        i += 1

    print(i, psi_zero_test(tensor, spin))

    return tensor


def kitaev_spin_one_half_ite_operator(tau):
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


def dimer_gas_operator(spin, phi):
    """Returns dimer gas operator (or variational ansatz) R for spin=1/2 or spin=1 Kitaev model."""

    zeta = np.zeros((2, 2, 2), dtype=complex)  # tau_tensor_{i j k}
    if spin == "1":
        zeta[0][0][0] = math.cos(phi)
        zeta[1][0][0] = zeta[0][1][0] = zeta[0][0][1] = math.sin(phi)
    elif spin == "1/2":
        # Dimer Gas:
        # zeta[0][0][0] = 1
        # zeta[1][0][0] = zeta[0][1][0] = zeta[0][0][1] = phi
        # Variational Ansatz:
        zeta[0][0][0] = math.cos(phi)
        zeta[1][0][0] = zeta[0][1][0] = zeta[0][0][1] = math.sin(phi)

    sx, sy, sz, one = constants.get_spin_operators(spin)
    d = one.shape[0]
    R = np.zeros((d, d, 2, 2, 2), dtype=complex)  # Q_LG_{s s' i j k}

    p = None
    if spin == "1":
        p = 1
    elif spin == "1/2":
        p = 0

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


def calculate_dimer_gas_profile(tensor_a, tensor_b, m, file_name='dimer_gas_profile.txt'):
    """Calculates energy for dimer gas state for values of variational parameter in fixed interval."""

    d = tensor_a.shape[0]  # physical dimension
    spin = None
    if d == 2:
        spin = "1/2"
    if d == 3:
        spin = "1"

    dim = tensor_a.shape[1] * 2  # virtual (bond) dimension

    lambdas = [np.ones((dim,), dtype=complex),
               np.ones((dim,), dtype=complex),
               np.ones((dim,), dtype=complex)]

    with open(file_name, 'w') as f:
        f.write('# %s S=%s model - 1st order String Gas\n' % (model, spin))
        f.write('# D=%d, m=%d\n' % (dim, m))
        f.write('# phi/pi\t\tEnergy\t\t\tConvergence\t\tCoarse-grain steps\n')

    for z in np.linspace(0, 0.5, num=100, endpoint=False):
        R = dimer_gas_operator(spin, z * math.pi)
        dimer_tensor_a = apply_gas_operator(tensor_a, R)
        dimer_tensor_b = apply_gas_operator(tensor_b, R)

        ctm = ctmrg.CTMRG(m, *honeycomb_expectation.export_to_ctmrg(dimer_tensor_a, dimer_tensor_b, lambdas, model))
        energy, delta, num_of_iter = ctm.ctmrg_iteration()

        # energy, delta, num_of_iter = honeycomb_expectation.coarse_graining_procedure(
        #    dimer_tensor_a, dimer_tensor_b, lambdas, m, model)

        print('Energy', - 3 * energy / 2, 'num_of_iter', num_of_iter)
        f = open(file_name, 'a')
        # f.write('%d\t\t%.15f\t%.15f\t%d\n' % (0, np.real(energy), np.real(mag_x), num_of_iter))
        # f.write('%.15f\t%.15f\t%d\n' % (z, - 3 * np.real(energy) / 2, num_of_iter))
        f.write('%.15f\t%.15f\t%.15f\t%d\n' % (z, - 3 * np.real(energy) / 2, delta, num_of_iter))
        # f.write('%d\t\t%.15f\t%d\n' % (0, np.real(energy), num_of_iter))
        f.close()


########################################################################################################################

model = "Kitaev"
# model = "Heisenberg"

spin = "1"  # implemented options so far: spin = "1", "1/2" for Kitaev model, spin = "1/2" for Heisenberg model
k = 1.
h = 0.E-14
# print('field', h)
D = 4  # max virtual (bond) dimension
m = 4  # bond dimension for coarse-graining (TRG or CTMRG); m should be at least D * D

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

# tensor_a = np.ones((d, xi, xi, xi)) / math.sqrt(d)
# tensor_b = np.ones((d, xi, xi, xi)) / math.sqrt(d)

# tensor_a = np.ones((d, xi, xi, xi), dtype=complex)
# tensor_b = np.ones((d, xi, xi, xi), dtype=complex)

tensor_a = np.zeros((d, xi, xi, xi), dtype=complex)
tensor_b = np.zeros((d, xi, xi, xi), dtype=complex)
lambdas = [np.array([1., ], dtype=complex), np.array([1., ], dtype=complex), np.array([1., ], dtype=complex)]

if model == "Kitaev":
    if spin == "1":
        # Spin=1 Kitaev model polarized state
        for i, x in enumerate(constants.mag_state_s1_kitaev):
            tensor_a[i][0][0][0] = x
            tensor_b[i][0][0][0] = x
    elif spin == "1/2":
        tensor_a = np.ones((d, xi, xi, xi), dtype=complex)
        tensor_b = np.ones((d, xi, xi, xi), dtype=complex)
        tensor_a = prepare_magnetized_state(tensor_a, spin)
        tensor_b = prepare_magnetized_state(tensor_b, spin)

if model == "Heisenberg":
    tensor_a[0][0][0][0] = 1.
    tensor_a[1][0][0][0] = 1.
    tensor_b[0][0][0][0] = 1.

# norm_a = np.max(np.abs(tensor_a))
# norm_b = np.max(np.abs(tensor_b))
# tensor_a /= norm_a
# tensor_b /= norm_b

# tensor_a = tensor_a / math.sqrt(np.real(calculate_tensor_norm(tensor_a)))
# tensor_b = tensor_b / math.sqrt(np.real(calculate_tensor_norm(tensor_b)))

if model == "Kitaev":

    Q_LG = constants.create_loop_gas_operator(spin)
    # tensor_a = np.einsum('s t i j k, t l m n->s i l j m k n', constants.Q_LG, tensor_a)
    # tensor_a = tensor_a.reshape((d, 2, 2, 2))
    tensor_a = apply_gas_operator(tensor_a, Q_LG)
    # tensor_b = np.einsum('s t i j k, t l m n->s i l j m k n', constants.Q_LG, tensor_b)
    # tensor_b = tensor_b.reshape((d, 2, 2, 2))
    tensor_b = apply_gas_operator(tensor_b, Q_LG)
    # tensor_a = tensor_a / math.sqrt(np.real(calculate_tensor_norm(tensor_a)))
    # tensor_b = tensor_b / math.sqrt(np.real(calculate_tensor_norm(tensor_b)))

    """
    lambdas = [np.array([1., 1.]) / math.sqrt(2),
               np.array([1., 1.]) / math.sqrt(2),
               np.array([1., 1.]) / math.sqrt(2)]
    """

    lambdas = [np.array([1., 1.], dtype=complex),
               np.array([1., 1.], dtype=complex),
               np.array([1., 1.], dtype=complex)]

    """
    # String-Gas state

    # phi1 = math.pi * 0.33
    # phi1 = math.pi * 0.32
    phi1 = math.pi * 0.275
    # phi1 = math.pi * 0.342
    R1 = dimer_gas_operator(spin, phi1)
    tensor_a = apply_gas_operator(tensor_a, R1)
    tensor_b = apply_gas_operator(tensor_b, R1)

    phi2 = math.pi * 0.22
    # phi2 = math.pi * 0.176
    R2 = dimer_gas_operator(spin, phi2)
    tensor_a = apply_gas_operator(tensor_a, R2)
    tensor_b = apply_gas_operator(tensor_b, R2)

    # For the case of 1st order dimer gas state:
    lambdas = [np.ones((4,), dtype=complex) / 2, 
               np.ones((4,), dtype=complex) / 2, 
               np.ones((4,), dtype=complex) / 2]

    # For the case of 2nd order dimer gas state:
    lambdas = [np.ones((8,), dtype=complex),
               np.ones((8,), dtype=complex),
               np.ones((8,), dtype=complex)]
    """

    calculate_dimer_gas_profile(tensor_a, tensor_b, m)

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


ctm = ctmrg.CTMRG(m, *honeycomb_expectation.export_to_ctmrg(tensor_a, tensor_b, lambdas, model))
energy, delta, num_of_iter = ctm.ctmrg_iteration()

# energy, delta, num_of_iter = honeycomb_expectation.coarse_graining_procedure(tensor_a, tensor_b, lambdas, m, model)
# print('Energy of the initial state', - 3 * energy / 2, 'num_of_iter', num_of_iter)
print('Energy of the initial state', - 3 * energy / 2, 'precision:', delta, 'num_of_iter', num_of_iter)
# print('Flux of the initial state', energy, 'num_of_iter', num_of_iter)


with open(file_name, 'w') as f:
    f.write('# %s S=%s model - ITE flow\n' % (model, spin))
    f.write('# D=%d, m=%d, tau=%.8E, h=%.14E\n' % (D, m, tau, h))
    f.write('# Iter\t\tEnergy\t\t\tConvergence\t\ttau\t\t\tCoarse-grain steps\n')
f = open(file_name, 'a')
# f.write('%d\t\t%.15f\t%.15f\t%d\n' % (0, np.real(energy), np.real(mag_x), num_of_iter))
# f.write('%d\t\t%.15f\t%d\n' % (0, - 3 * np.real(energy) / 2, num_of_iter))
f.write('%d\t\t%.15f\t%.15f\t%.15f\t%d\n' % (0, - 3 * np.real(energy) / 2, delta, 0, num_of_iter))
# f.write('%d\t\t%.15f\t%d\n' % (0, np.real(energy), num_of_iter))
f.close()

energy_old = -1

lambdas_memory = copy.deepcopy(lambdas)

u_gates = None
if model == "Kitaev" and spin == "1":
    u_gates = [exp_ham.reshape(d, d, d, d) for exp_ham in kitaev_spin_one_ite_operator(tau)]  # analytic form
else:
    # Heisenberg and spin=1/2 Kitaev
    H = construct_hamiltonian(h, spin, k)
    u_gates = np.array([construct_ite_operator(tau, hamiltonian).reshape(d, d, d, d) for hamiltonian in H])


j = 0  # ITE-step index

while tau >= tau_final and (j * refresh < 2600):

    for i in tqdm(range(refresh)):
        tensor_a, tensor_b, lambdas = update_step(tensor_a, tensor_b, lambdas, u_gates)

    print('iter', (j + 1) * refresh)
    print('tau', tau)
    print(lambdas[0][:12])
    print(lambdas[1][:12])
    print(lambdas[2][:12])

    """
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
    energy, delta, num_of_iter = \
        honeycomb_expectation.coarse_graining_procedure(tensor_a_copy, tensor_b_copy, lambdas_copy, m, model)
    """

    ctm = ctmrg.CTMRG(m, *honeycomb_expectation.export_to_ctmrg(tensor_a, tensor_b, lambdas, model))
    energy, delta, num_of_iter = ctm.ctmrg_iteration()
    # energy, delta, num_of_iter = \
    #    honeycomb_expectation.coarse_graining_procedure(tensor_a, tensor_b, lambdas, m, model)

    energy = - 3 * energy / 2
    # energy = energy / 4

    # print('# ITE flow iter:', (j + 1) * refresh, 'energy:', energy, 'mag_x:', mag_x, 'num_of_iter:', num_of_iter)
    # print('# ITE flow iter:', (j + 1) * refresh, 'energy:', energy, 'num_of_iter:', num_of_iter)
    print('# ITE flow iter:', (j + 1) * refresh, 'energy:', energy, 'delta:', delta, 'num_of_iter:', num_of_iter)

    f = open(file_name, 'a')
    # f.write('%d\t\t%.15f\t%.15f\t%d\n' % ((j + 1) * refresh, np.real(energy), np.real(mag_x), num_of_iter))
    f.write('%d\t\t%.15f\t%.15f\t%.15f\t%d\n' % ((j + 1) * refresh, np.real(energy), delta, tau, num_of_iter))
    f.close()

    s1 = abs_list_difference(lambdas[0], lambdas_memory[0])
    s2 = abs_list_difference(lambdas[1], lambdas_memory[1])
    s3 = abs_list_difference(lambdas[2], lambdas_memory[2])
    lambdas_memory = copy.deepcopy(lambdas)
    print(s1 + s2 + s3)
    # if s1 < 1.E-6 and s2 < 1.E-6 and s3 < 1.E-6:
    if s1 < 1.E-11 and s2 < 1.E-11 and s3 < 1.E-11:
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
