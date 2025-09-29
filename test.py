import numpy as np


z0 = 36.222512330899455
z = 7.6414264952211
x = np.tan(z) - np.sqrt(z0**2 - z**2) / z
print(x)