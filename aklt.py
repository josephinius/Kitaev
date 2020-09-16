import numpy as np
from ncon import ncon
import ctmrg
import honeycomb_expectation
from constants import sy12 as SY

a0 = 1
a2 = np.sqrt(6)
a1 = np.sqrt(3 / 2)

d = 5
dim = 2

aklt_state = np.zeros([d, dim, dim, dim, dim])

aklt_state[0, 0, 0, 0, 0] = aklt_state[4, 1, 1, 1, 1] = a2
aklt_state[1, 0, 0, 0, 1] = aklt_state[1, 0, 0, 1, 0] = aklt_state[1, 0, 1, 0, 0] = aklt_state[1, 1, 0, 0, 0] = a1
aklt_state[3, 1, 1, 1, 0] = aklt_state[3, 1, 1, 0, 1] = aklt_state[3, 1, 0, 1, 1] = aklt_state[3, 0, 1, 1, 1] = a1
aklt_state[2, 0, 0, 1, 1] = aklt_state[2, 0, 1, 1, 0] = aklt_state[2, 1, 1, 0, 0] = aklt_state[2, 1, 0, 0, 1] = \
    aklt_state[2, 0, 1, 0, 1] = aklt_state[2, 1, 0, 1, 0] = a0

string = 1j * SY

aklt_state = ncon([aklt_state, string, string],
                  [[-1, -2, -3, 1, 2], [1, -4], [2, -5]])
aklt_state /= np.sqrt(6)

aklt_weight = ncon(
    [aklt_state,
     aklt_state],
    [[1, -1, -3, -5, -7],
     [1, -2, -4, -6, -8]]
)

# Exporting aklt_weight to CTMRG

w = np.transpose(aklt_weight, (0, 1, 6, 7, 4, 5, 2, 3))
corners, tms = honeycomb_expectation.weight_to_ctm(w)

# Running CTMRG

w = w.reshape((dim * dim, dim * dim, dim * dim, dim * dim))

m = 32

ctm = ctmrg.CTMRG(m, w, corners, tms, w, algorithm='Corboz')
e, delta, correlation_length, num_of_iter = ctm.ctmrg_iteration()

print('e', e)
print('delta', delta)
print('correlation length', correlation_length)
print('number of iterations', num_of_iter)
