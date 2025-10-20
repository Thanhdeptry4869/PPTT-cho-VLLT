import numpy as np
from matplotlib import pyplot as plt

data = np.loadtxt("results_psi.dat")

x = data[:, 0]
psi = data[:, 1]

plt.plot(x, psi)
plt.xlabel("x")
plt.ylabel("ψ(x)")
plt.show()