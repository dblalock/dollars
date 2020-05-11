
import numpy as np
gen = np.random.default_rng()

N = int(1e6)

for sigma in np.arange(1, 601) / 1000.:
    X = np.maximum(1e-20, gen.laplace(loc=1., scale=sigma/np.sqrt(2), size=N))
    print("{:.3f}\t{:.4f}\t%".format(
        sigma, (np.exp(np.log(X).mean()) - 1) * 100))

