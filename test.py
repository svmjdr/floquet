import qutip
import numpy as np
from qutip.qobj import Qobj


a = Qobj([[1,3],[97,4]])

eigvals, eigvecs = Qobj(a).eigenstates()
kernel_index = np.argmin(np.abs(eigvals))

pr = eigvals[np.isclose(eigvals,eigvals[kernel_index])]

print(pr)
