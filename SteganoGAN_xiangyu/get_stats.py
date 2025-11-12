import numpy as np
import os


if __name__ == "__main__":
    for bit in range(1, 5):
        fname = os.path.join(f"{bit}_bits", "err.txt")
        with open(fname) as f:
            err = [float(x.strip()) * 100.0 for x in f.readlines()]
        print(f"{bit} bits: mean {np.mean(err):0.2f}, std {np.std(err):0.2f}")
