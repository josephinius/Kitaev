import numpy as np
from ncon import ncon
# import time
# from tqdm import tqdm
# import constants
import honeycomb_expectation
import ctmrg

"""

    eps_state_{ijk}
 
     k
       \
        o -- i
       /
     j

    projector_{aij}
    
      i -- O -- j
           |
           a
             
"""

levi_civita = np.zeros((3, 3, 3))
levi_civita[0, 1, 2] = levi_civita[1, 2, 0] = levi_civita[2, 0, 1] = 1
levi_civita[2, 1, 0] = levi_civita[0, 2, 1] = levi_civita[1, 0, 2] = -1

eps_state = np.zeros((3, 3, 3))

for i in range(3):
    for j in range(3):
        for k in range(3):
            eps_state[i, j, k] += levi_civita[i, j, k]

eps_state[2, 2, 2] += 1


# p = |0>(<02| + <20|) + |1>(<12| + <21|)

projector = np.zeros((2, 3, 3))
projector[0, 0, 2] = projector[0, 2, 0] = projector[1, 1, 2] = projector[1, 2, 1] = 1

rvb_state = ncon(
    [eps_state, projector, eps_state, projector, projector],
    [[1, -4, -5], [-1, 1, 2], [2, 3, 4], [-2, 3, -6], [-3, 4, -7]]
)

rvb_weight = ncon(
    [rvb_state,
     rvb_state],
    [[1, 2, 3, -1, -3, -5, -7],
     [1, 2, 3, -2, -4, -6, -8]]
)

# Exporting rvb_weight to CTMRG

w = np.transpose(rvb_weight, (0, 1, 6, 7, 4, 5, 2, 3))
corners, tms = honeycomb_expectation.weight_to_ctm(w)

# Running CTMRG

m = 64

w = w.reshape((3 * 3, 3 * 3, 3 * 3, 3 * 3))

ctm = ctmrg.CTMRG(m, w, corners, tms, w, algorithm='Corboz')
e, delta, correlation_length, num_of_iter = ctm.ctmrg_iteration()

print('e', e)
print('delta', delta)
print('correlation length', correlation_length)
print('number of iterations', num_of_iter)
